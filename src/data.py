"""데이터 로딩 공용 함수 (탐색·학습·평가 노트북에서 공유)."""
import json
from collections import Counter

import pandas as pd


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


def stt_text(segments, start, end):
    """[start, end) 구간과 겹치는 STT 세그먼트 텍스트를 이어 붙인다."""
    # ponytail: 시간 단위가 문장과 STT 간에 같다고 가정. 다르면(초 vs ms) 여기서 환산.
    return " ".join(
        str(s.get("value", ""))
        for s in segments
        if float(s["startTime"]) < end and float(s["endTime"]) > start
    ).strip()


def dialect_sentences(doc):
    """방언 라벨링 json 하나 → 문장 단위 dict 리스트."""
    spk = speakers(doc)
    by_id = {s.get("id"): s for s in spk}
    provinces = {s.get("residenceProvince") for s in spk}
    ann = doc.get("annotation") or {}
    emotion = {e.get("sentenceId"): e.get("tagType") for e in _as_list(ann.get("emotions"))}
    n_inton = Counter(i.get("sentenceId") for i in _as_list(ann.get("intonations")))
    segs = _as_list((doc.get("stt") or {}).get("segments"))

    rows = []
    for s in doc["transcription"]["sentences"]:
        sid = s["sentenceId"]
        start, end = float(s["startTime"]), float(s["endTime"])
        # 문장에 speakerId가 없고 화자가 여럿이면 화자를 특정하지 않는다 (잘못 붙이는 것보다 낫다)
        who = by_id.get(s.get("speakerId")) or (spk[0] if len(spk) == 1 else {})
        rows.append({
            "sentenceId": sid,
            "startTime": start,
            "endTime": end,
            "dialect": s.get("dialect"),
            "standard": s.get("standard"),
            "speakerId": who.get("id"),
            "birthYear": who.get("birthYear"),
            "province": who.get("residenceProvince")
            or (next(iter(provinces)) if len(provinces) == 1 else None),
            "emotion": emotion.get(sid),
            "n_intonations": n_inton.get(sid, 0),
            "stt": stt_text(segs, start, end),
        })
    return rows


if __name__ == "__main__":
    doc = {
        "speaker": [
            {"id": "A", "birthYear": 1955, "residenceProvince": "경상북도"},
            {"id": "B", "birthYear": 1960, "residenceProvince": "경상북도"},
        ],
        "transcription": {"sentences": [
            {"sentenceId": 1, "startTime": 0.0, "endTime": 2.0, "dialect": "밥 뭇나", "standard": "밥 먹었니", "speakerId": "B"},
            {"sentenceId": 2, "startTime": 2.0, "endTime": 4.0, "dialect": "어", "standard": "어"},
        ]},
        "annotation": {
            "emotions": [{"sentenceId": 1, "tagType": "무관"}],
            "intonations": [{"sentenceId": 1}, {"sentenceId": 1}],
        },
        "stt": {"segments": [
            {"startTime": 0.1, "endTime": 1.0, "value": "밥"},
            {"startTime": 1.0, "endTime": 1.9, "value": "뭇나"},
            {"startTime": 2.5, "endTime": 3.0, "value": "어"},
        ]},
    }
    r1, r2 = dialect_sentences(doc)
    assert r1["speakerId"] == "B" and r1["birthYear"] == 1960
    assert r1["emotion"] == "무관" and r1["n_intonations"] == 2 and r1["stt"] == "밥 뭇나"
    assert r2["speakerId"] is None and r2["province"] == "경상북도"
    assert r2["emotion"] is None and r2["n_intonations"] == 0 and r2["stt"] == "어"
    print("ok")
