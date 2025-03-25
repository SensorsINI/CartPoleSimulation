# config.py

class Config:
    """User-defined configuration."""
    def __init__(self) -> None:
        self.data_folder = r"../../SI_Toolkit_ASF/Experiments/Train_NN_what_new/"
        self.state_columns = [
            "angle", "angleD", "angle_cos", "angle_sin",
            "position", "positionD", "target_position", "target_equilibrium"
        ]
        self.teacher_control_columns = ["Q_calculated_offline"]
        self.student_control_columns = ["Q_calculated_offline_NN"]
        self.relative_error_cap = 10.0
        self.n_starting_steps = 50
        self.norm_file_path = (
            '../../SI_Toolkit_ASF/Experiments/GRU-7IN-32H1-32H2-1OUT-3/NI_2024-08-17_22-25-35.csv'
        )
