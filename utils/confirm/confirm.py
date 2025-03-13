def confirm_action(prompt):
    choice = input(prompt).strip().lower()
    if choice in ["y", "yes"]:
        print("Confirmed.")
        return True
    elif choice in ["n", "no"]:
        print("Cancelled.")
        return False
    else:
        print(f"Invalid input: '{choice}'. Operation cancelled.")
        return False


def confirm_call(prompt):
    def confirm_func(func):
        def wrapper(*args, **kwargs):
            if not confirm_action(prompt):
                print("Operation aborted.")
                return None
            return func(*args, **kwargs)

        return wrapper

    return confirm_func
