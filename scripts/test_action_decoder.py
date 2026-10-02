import numpy as np

from dhelm_opf.models.action_decoder import ActionDecoder


def main() -> None:

    decoder = ActionDecoder(
        generator_pmin=(0.0, 0.0, 0.0, 0.0),
        generator_pmax=(5.0, 5.0, 5.0, 5.0),
    )

    # All -1 -> lower physical bounds
    action_min = np.full(8, -1.0)

    decoded_min = decoder.decode(action_min)

    print("Minimum normalized action:")
    print(decoded_min)

    assert decoded_min.voltage_setpoints == (
        0.9,
        0.9,
        0.9,
        0.9,
    )

    assert decoded_min.generator_p_setpoints == (
        0.0,
        0.0,
        0.0,
        0.0,
    )

    # All +1 -> upper physical bounds
    action_max = np.full(8, 1.0)

    decoded_max = decoder.decode(action_max)

    print("\nMaximum normalized action:")
    print(decoded_max)

    assert decoded_max.voltage_setpoints == (
        1.1,
        1.1,
        1.1,
        1.1,
    )

    assert decoded_max.generator_p_setpoints == (
        5.0,
        5.0,
        5.0,
        5.0,
    )

    # Zero -> midpoint of physical bounds
    action_zero = np.zeros(8)

    decoded_zero = decoder.decode(action_zero)

    print("\nZero normalized action:")
    print(decoded_zero)

    assert decoded_zero.voltage_setpoints == (
        1.0,
        1.0,
        1.0,
        1.0,
    )

    assert decoded_zero.generator_p_setpoints == (
        2.5,
        2.5,
        2.5,
        2.5,
    )

    print("\nAll action-decoder tests passed.")


if __name__ == "__main__":
    main()
