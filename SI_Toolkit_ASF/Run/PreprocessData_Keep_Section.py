from SI_Toolkit.data_preprocessing import transform_dataset

get_files_from = 'SI_Toolkit_ASF/Experiments/CPP_MPCswingups_before_23_04_2025_min/Recordings/'
save_files_to = 'SI_Toolkit_ASF/Experiments/CPP_MPCswingups_before_23_04_2025_min_40/Recordings/'
section_to_keep = [0.0, 0.4]
# section_to_keep = [141, -1]
mode = 'percent'  # 'lines' or 'percent'

transform_dataset(get_files_from, save_files_to, transformation='keep_section', mode=mode,
                    section_to_keep=section_to_keep)

