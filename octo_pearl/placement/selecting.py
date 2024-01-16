from typing import List

from octo_pearl.placement.utils import gpt4, read_file_to_string

USR_PATH = "octo_pearl/placement/selecting_user_template.txt"
SYS_PATH = "octo_pearl/placement/selecting_system_template.txt"


def select_best_tag(
    filtered_tags: List[str], object_to_place: str, api_key: str = ""
) -> str:
    user_template = read_file_to_string(USR_PATH).format(object=object_to_place)
    user_prompt = user_template + "\n".join(filtered_tags)
    system_prompt = read_file_to_string(SYS_PATH)
    return gpt4(user_prompt, system_prompt, api_key=api_key)
