def print_runtime(text: str, runtime: float) -> None:
    """
    Print runtime in seconds.
    :param text: Message to print to the console indicating what was measured.
    :param runtime: The runtime in seconds.
    :return: None.
    """
    print("\n--- {} runtime: {} seconds ---".format(text, runtime))
