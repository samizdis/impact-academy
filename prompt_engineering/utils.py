def scale_score(baseline, final, max_score=0.6):
    if final < baseline:
        return 0
    return (max_score / (1 - baseline)) * (final - baseline)


if __name__ == '__main__':
    assert scale_score(0, 0) == 0
    assert scale_score(0, 1) == 0.6
    assert scale_score(0, 0.99) < 0.6
    assert scale_score(0, 0.01) > 0
    assert scale_score(0.5, 0.6) > 0 and scale_score(0.5, 0.6) < 0.6