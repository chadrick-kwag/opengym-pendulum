
def normalize_state(state):
    # range: [-2,2] -> [-1,1]


    return state/2


def denormalize_action(action_val):

    return action_val * 2
