import pytest

from chunking import chunk_text


def words(n):
    return " ".join(f"w{i}" for i in range(n))


def test_short_text_is_one_chunk():
    assert chunk_text(words(10), chunk_size=500, overlap=50) == [words(10)]


def test_empty_text_gives_no_chunks():
    assert chunk_text("", chunk_size=500, overlap=50) == []


def test_chunks_overlap_by_the_requested_number_of_words():
    chunks = chunk_text(words(25), chunk_size=10, overlap=3)
    first, second = chunks[0].split(), chunks[1].split()
    assert len(first) == 10
    assert first[-3:] == second[:3]


def test_every_word_is_covered():
    text = words(1234)
    chunks = chunk_text(text, chunk_size=500, overlap=50)
    covered = set(" ".join(chunks).split())
    assert covered == set(text.split())


def test_no_redundant_tail_chunk_when_text_fits_exactly():
    # 500 words with chunk_size=500 must not produce a second chunk that is
    # only a copy of the last 50 words.
    assert len(chunk_text(words(500), chunk_size=500, overlap=50)) == 1


@pytest.mark.parametrize("chunk_size,overlap", [(0, 0), (10, 10), (10, -1)])
def test_invalid_settings_are_rejected(chunk_size, overlap):
    with pytest.raises(ValueError):
        chunk_text(words(20), chunk_size=chunk_size, overlap=overlap)
