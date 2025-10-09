from PIL import Image
from api.inference import mock_predict


def test_mock_predict_shapes_and_ranges():
    img = Image.new("RGB", (320, 240), color=(180, 180, 180))
    label, conf, boxes = mock_predict(img)

    assert label in {"HasStone", "NoStone"}
    assert 0.0 <= conf <= 1.0

    if label == "HasStone":
        assert len(boxes) >= 1
        for (x, y, w, h) in boxes:
            assert x >= 0 and y >= 0 and w > 0 and h > 0
    else:
        # NoStone — дозволяємо 0 боксів
        assert len(boxes) == 0
