from subtitler.captions import Word, group_words, write_ass, write_srt


def _words(*items):
    return [Word(s, e, t) for s, e, t in items]


def test_splits_on_max_words():
    words = _words(*[(i, i + 0.5, f"w{i}") for i in range(6)])
    caps = group_words(words, max_words=4, max_gap=10)
    assert [c.text for c in caps] == ["w0 w1 w2 w3", "w4 w5"]


def test_splits_on_pause_and_punctuation():
    words = _words((0, 0.4, "Hello."), (0.5, 0.9, "How"), (0.9, 1.2, "are"), (3.0, 3.3, "you"))
    caps = group_words(words, max_words=10)
    assert [c.text for c in caps] == ["Hello.", "How are", "you"]
    assert caps[1].start == 0.5 and caps[1].end == 1.2


def test_srt_and_ass_format(tmp_path):
    caps = group_words(_words((0, 1.25, "hi"), (61.5, 62.0, "there")), max_gap=0.5)
    srt = write_srt(caps, tmp_path / "a.srt").read_text()
    assert "00:00:00,000 --> 00:00:01,250" in srt
    assert "00:01:01,500 --> 00:01:02,000" in srt
    ass = write_ass(caps, tmp_path / "a.ass", 1080, 1920).read_text()
    assert "PlayResY: 1920" in ass
    assert "Dialogue: 0,0:01:01.50,0:01:02.00,Reels,,0,0,0,,THERE" in ass