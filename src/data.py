"""데이터 로딩 공용 함수 (탐색·전처리·학습 노트북에서 공유)."""
import json
from bisect import bisect_right
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

NAN = float("nan")
DEFAULT_RECORD_YEAR = 2022  # audio.recordDate가 없을 때 쓰는 녹음 연도


def read_csv_any(path):
    """AI Hub csv는 utf-8과 cp949가 섞여 있어 둘 다 시도한다."""
    try:
        return pd.read_csv(path, encoding="utf-8-sig")
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="cp949")


def load_json(path):
    with open(path, encoding="utf-8-sig") as f:
        return json.load(f)


def _as_list(x):
    if x is None:
        return []
    return x if isinstance(x, list) else [x]


def speakers(doc):
    """speaker 필드는 1인이면 dict, 2인발화면 list일 수 있다 → 항상 list로."""
    return _as_list(doc.get("speaker"))


def hms(t):
    """'HH:MM:SS.mmm' → 초."""
    h, m, s = str(t).split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def record_year(doc):
    y = str((doc.get("audio") or {}).get("recordDate") or "")[:4]
    return int(y) if y.isdigit() else DEFAULT_RECORD_YEAR


def f0_stats(values):
    """문장의 F0(Hz) 리스트 → 평균·표준편차·5~95 백분위 범위(반음)·유효 개수. 0 이하·NaN 제거."""
    f = pd.to_numeric(pd.Series(_as_list(values), dtype=object), errors="coerce").to_numpy(float)
    f = f[np.isfinite(f) & (f > 0)]
    if len(f) == 0:
        return {"f0_mean": NAN, "f0_std": NAN, "f0_range_st": NAN, "f0_count": 0}
    p5, p95 = np.percentile(f, [5, 95])
    return {"f0_mean": f.mean(), "f0_std": f.std(), "f0_range_st": 12 * np.log2(p95 / p5), "f0_count": len(f)}


# ----- H 억양 태그 → 문장 매핑 -----
# annotation.intonations에는 sentenceId가 없고 '파일 전체 전사' 기준 글자 위치만 있다.
# 어떤 문자열이 기준인지 몰라서 (문장 필드, 문장 사이 구분자, 위치 시작값) 조합을 모두 시도한다.
H_RULES = [(field, sep, base)
           for field in ("dialect", "pronunciation", "standard")
           for sep in ("", " ", "\n")
           for base in (0, 1)]


def h_map(doc, rule):
    """H 태그마다 문장 인덱스(매핑 실패는 None)를 돌려준다.

    rule=(field, sep, base): 문장들의 field 텍스트를 sep으로 이어 붙인 문자열이 전체 전사이고,
    글자 위치는 base(0/1)부터 센다고 가정. 성공 = 태그 구간이 한 문장 안에 들어가고 공백에서 시작하지 않음.
    """
    field, sep, base = rule
    texts = [str(s.get(field) or "") for s in doc["transcription"]["sentences"]]
    starts, pos = [], 0
    for t in texts:
        starts.append(pos)
        pos += len(t) + len(sep)
    full = sep.join(texts)
    out = []
    for tag in _as_list((doc.get("annotation") or {}).get("intonations")):
        if tag.get("tagType") != "H":
            continue
        try:
            b = int(tag["transcriptionBeginInFile"]) - base
            e = int(tag["transcriptionEndInFile"]) - base
        except (KeyError, TypeError, ValueError):
            out.append(None)
            continue
        i = bisect_right(starts, b) - 1
        end_i = starts[i] + len(texts[i]) if i >= 0 else -1
        ok = 0 <= b < end_i and b <= e <= end_i and not full[b].isspace()
        out.append(i if ok else None)
    return out


def choose_h_rule(docs, min_rate=0.9):
    """H_RULES별 매핑 성공률을 출력하고 가장 높은 규칙을 반환. min_rate 미만이면 None(→ h_count NaN)."""
    rates = {}
    for rule in H_RULES:
        res = [i for d in docs for i in h_map(d, rule)]
        rates[rule] = sum(i is not None for i in res) / max(len(res), 1)
    n_tags = sum(len(h_map(d, H_RULES[0])) for d in docs)
    print(f"json {len(docs)}개, H 태그 {n_tags}개")
    for (field, sep, base), r in sorted(rates.items(), key=lambda kv: -kv[1]):
        print(f"  field={field:<13} sep={sep!r:<5} base={base}  성공률 {r:6.1%}")
    best = max(rates, key=rates.get)
    if rates[best] < min_rate:
        print(f"최고 성공률 {rates[best]:.1%} < {min_rate:.0%} → 매핑 불가, h_count는 NaN으로 둡니다")
        return None
    print("사용 규칙:", best, "(상위 규칙끼리 성공률이 비슷하면 결과 해석 시 주의)")
    return best


def dialect_sentences(doc, h_rule=None):
    """방언 라벨링 json 하나 → 문장 단위 dict 리스트. h_rule이 None이면 h_count는 NaN."""
    spk = speakers(doc)
    by_id = {s.get("speakerId"): s for s in spk}
    provinces = {s.get("residenceProvince") for s in spk}
    year = record_year(doc)
    ann = doc.get("annotation") or {}
    emotion = {float(e["sentenceId"]): e.get("tagType")
               for e in _as_list(ann.get("emotions")) if e.get("sentenceId") is not None}
    segs = [(hms(s["startTime"]), hms(s["endTime"]), str(s.get("value", "")))
            for s in _as_list((doc.get("stt") or {}).get("segments"))]
    h = Counter(h_map(doc, h_rule)) if h_rule else None
    file = Path(str(doc.get("fileName", ""))).stem

    rows = []
    for i, s in enumerate(doc["transcription"]["sentences"]):
        sid = float(s["sentenceId"])
        start, end = hms(s["startTime"]), hms(s["endTime"])
        # 문장 speakerId로 화자를 찾고, 못 찾으면 1인 파일일 때만 그 화자로 (잘못 붙이는 것보다 비우는 게 낫다)
        who = by_id.get(s.get("speakerId")) or (spk[0] if len(spk) == 1 else {})
        try:
            age = year - int(who.get("birthYear"))
        except (TypeError, ValueError):
            age = NAN
        rows.append({
            "region": who.get("residenceProvince") or (next(iter(provinces)) if len(provinces) == 1 else None),
            "file": file,
            "sentence_id": sid,
            "speaker_id": s.get("speakerId") or who.get("speakerId"),
            "gender": who.get("gender"),
            "age": age,
            "start": start,
            "end": end,
            "duration": end - start,
            "dialect": s.get("dialect"),
            "standard": s.get("standard"),
            "stt": " ".join(v for a, b, v in segs if a < end and b > start).strip(),
            "emotion_tag": emotion.get(sid),
            **f0_stats(s.get("intonations")),
            "h_count": h.get(i, 0) if h is not None else NAN,
        })
    return rows


# ----- 감정 데이터 -----
EMO_VOTE_COLS = [f"{i}번 감정" for i in range(1, 6)]  # 세기 컬럼은 '4번감정세기'만 띄어쓰기 없음 (여기선 안 씀)


def majority(votes, situation=None):
    """5명 감정 → (다수결 label, 다수결 표 수/5). 동점이면 상황 값이 후보에 있으면 상황, 아니면 후보 중 먼저 나온 것."""
    votes = [str(v).strip().lower() for v in votes if pd.notna(v) and str(v).strip()]
    if not votes:
        return None, 0.0
    c = Counter(votes)
    top = max(c.values())
    cands = [v for v in dict.fromkeys(votes) if c[v] == top]
    sit = str(situation).strip().lower() if pd.notna(situation) else None
    return (sit if sit in cands else cands[0]), top / 5


def load_emotion(csv_path, wav_dir):
    """감정 csv(cp949) → label·agreement·wav_path 컬럼 추가."""
    df = pd.read_csv(csv_path, encoding="cp949")
    res = [majority(v, s) for v, s in zip(df[EMO_VOTE_COLS].values.tolist(), df["상황"])]
    df["label"] = [r[0] for r in res]
    df["agreement"] = [r[1] for r in res]
    df["wav_path"] = [f"{wav_dir}/{w}.wav" for w in df["wav_id"]]
    return df


def cer(ref, hyp):
    """문자 오류율: 공백 뺀 글자 기준 편집거리 / 정답 글자 수."""
    r, h = "".join(str(ref).split()), "".join(str(hyp).split())
    prev = list(range(len(h) + 1))
    for i, rc in enumerate(r, 1):
        cur = [i]
        for j, hc in enumerate(h, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (rc != hc)))
        prev = cur
    return prev[-1] / max(len(r), 1)


if __name__ == "__main__":
    doc = {
        "fileName": "DGS0001.wav",
        "audio": {"recordDate": "2021-08-01"},
        "speaker": [
            {"speakerId": "A", "gender": "여", "birthYear": 1955, "residenceProvince": "gs"},
            {"speakerId": "B", "gender": "남", "birthYear": "1960", "residenceProvince": "gs"},
        ],
        "transcription": {"sentences": [
            {"sentenceId": 1.0, "speakerId": "B", "startTime": "00:00:00.000", "endTime": "00:00:02.000",
             "dialect": "밥 뭇나", "standard": "밥 먹었니", "intonations": [100, 0, None, "nan", 200, 150]},
            {"sentenceId": 2.0, "startTime": "00:00:02.000", "endTime": "00:01:04.500",
             "dialect": "어 뭇다", "standard": "어 먹었다", "intonations": []},
        ]},
        "annotation": {
            "emotions": [{"sentenceId": 1.0, "tagType": "neutral"}],
            # 기준 문자열 "밥 뭇나 어 뭇다" (dialect, sep=" ", base=0): '뭇나'=2~4, '뭇다'=7~9
            "intonations": [
                {"transcriptionBeginInFile": 2, "transcriptionEndInFile": 4, "tagType": "H"},
                {"transcriptionBeginInFile": 7, "transcriptionEndInFile": 9, "tagType": "H"},
            ],
        },
        "stt": {"segments": [
            {"startTime": "00:00:00.100", "endTime": "00:00:01.000", "value": "밥"},
            {"startTime": "00:00:01.000", "endTime": "00:00:01.900", "value": "뭇나"},
            {"startTime": "00:00:02.500", "endTime": "00:00:03.000", "value": "어"},
        ]},
    }
    assert hms("01:02:03.500") == 3723.5
    rule = ("dialect", " ", 0)
    assert h_map(doc, rule) == [0, 1]
    assert h_map(doc, ("dialect", "", 0)) == [0, None]  # 구분자 없으면 두 번째 태그가 경계를 넘음
    r1, r2 = dialect_sentences(doc, rule)
    assert r1["file"] == "DGS0001" and r1["speaker_id"] == "B" and r1["age"] == 61 and r1["gender"] == "남"
    assert r1["emotion_tag"] == "neutral" and r1["stt"] == "밥 뭇나" and r1["h_count"] == 1
    assert r1["f0_count"] == 3 and abs(r1["f0_mean"] - 150) < 1e-9 and r1["f0_range_st"] > 0
    assert r2["speaker_id"] is None and r2["region"] == "gs" and r2["age"] != r2["age"]  # NaN
    assert r2["emotion_tag"] is None and r2["stt"] == "어" and r2["duration"] == 62.5 and r2["f0_count"] == 0
    assert dialect_sentences(doc)[0]["h_count"] != dialect_sentences(doc)[0]["h_count"]  # 규칙 없으면 NaN
    assert choose_h_rule([doc]) == ("dialect", " ", 0)

    assert majority(["angry", "angry", "sadness", "neutral", "angry"], "sadness") == ("angry", 0.6)
    assert majority(["Angry", "sadness", "sadness", "angry", "fear"], "sadness") == ("sadness", 0.4)
    assert majority(["fear", "sadness", "sadness", "fear", "angry"], "happiness") == ("fear", 0.4)
    assert majority([None, float("nan")]) == (None, 0.0)

    assert cer("밥 뭇나", "밥뭇나") == 0 and cer("abcd", "abxd") == 0.25 and cer("ab", "") == 1
    print("ok")
