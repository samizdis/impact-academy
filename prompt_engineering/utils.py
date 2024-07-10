def scale_score(baseline, final):
    return max(final - baseline, 0.0)


if __name__ == '__main__':
    assert scale_score(0, 0.44) == 0.72
    assert scale_score(0, 1) == 1.0
    assert scale_score(0, 0) == 0.5
    assert scale_score(0, 0.01) == 0.01
    assert scale_score(0.5, 0.6) > 0 and scale_score(0.5, 0.6) < 1.0