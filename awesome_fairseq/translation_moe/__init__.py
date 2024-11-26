from awesome_fairseq.translation import Translation, register_translation_tasks

register_translation_moe_tasks = register_translation_tasks


class TranslationMOE(Translation):
    task_group: str = "translation_moe"
