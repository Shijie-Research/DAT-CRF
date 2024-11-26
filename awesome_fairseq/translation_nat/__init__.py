from awesome_fairseq.translation import Translation, register_translation_tasks

register_nat_tasks = register_translation_tasks


class NATranslation(Translation):
    task_group: str = "translation_nat"
