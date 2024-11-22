from CartPole import Generate_Random_Trace_Function


def add_new_target_position(df, target_position_config, new_target_position_variable_name, **kwargs):

    time = (df['time'] - df['time'].iloc[0]).to_numpy()
    length_of_experiment = time[-1]
    target_position_generating_function = Generate_Random_Trace_Function(
        length_of_experiment=length_of_experiment,
        **target_position_config
    )

    df[new_target_position_variable_name] = target_position_generating_function(time)

    return df


def flip_target_equilibrium(df, new_target_equilibrium_variable_name, **kwargs):

    df[new_target_equilibrium_variable_name] = -df['target_equilibrium']

    return df
