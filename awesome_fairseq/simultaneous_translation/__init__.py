from awesome_fairseq.translation import Translation, register_translation_tasks

register_simul_translation_tasks = register_translation_tasks


class SimulTranslation(Translation):
    task_group: str = "simul_translation"
