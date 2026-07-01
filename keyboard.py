
KeyboardRowItem = str | float

keyboard_matrix: list[list[KeyboardRowItem]] = [
    ["q", "w", "e", "r", "t", "y", "u", "i", "o", "p"],
    [0.5, "a", "s", "d", "f", "g", "h", "j", "k", "l", 0.5],
    [1.5, "z", "x", "c", "v", "b", "n", "m", 1.5],
]


def isfloat(s: str) -> float | None:
    try:
        x = float(s)
    except ValueError:
        return None

    if x == 0.0:
        raise ValueError
    return x


def get_keyboard_coordinates(keyboard_matrix) -> dict[str, tuple[float, int]]:
    coords: dict[str, tuple[float, int]] = {}
    for y, row in enumerate(keyboard_matrix):
        x_offset: float = 0.0
        for item in row:
            if offset := isfloat(item):
                x_offset += offset
            else:
                coords[item] = (x_offset, y)
                x_offset += 1.0
    return coords
