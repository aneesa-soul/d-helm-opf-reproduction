from dhelm_opf.models.action_config import DEFAULT_ACTION_CONFIG


def main() -> None:
    config = DEFAULT_ACTION_CONFIG

    print(config.describe())
    print()

    print("Action dimension:", config.action_dim)
    print("Voltage action indices:", config.voltage_action_indices)
    print("Generator action indices:", config.generator_action_indices)

    assert config.action_dim == 8
    assert len(config.voltage_action_indices) == 4
    assert len(config.generator_action_indices) == 4


if __name__ == "__main__":
    main()

