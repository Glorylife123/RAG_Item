from __future__ import annotations

import csv
from pathlib import Path


class DrugInteractionTool:
    name = "get_drug_interaction"
    description = "查询两个药物之间是否存在常见相互作用。参数: drug_a, drug_b。"

    def __init__(self, csv_path: str | Path) -> None:
        self.csv_path = Path(csv_path)
        self.rows = self._load_rows()

    def run(self, arguments: dict) -> str:
        drug_a = str(arguments.get("drug_a") or arguments.get("a") or "").strip().lower()
        drug_b = str(arguments.get("drug_b") or arguments.get("b") or "").strip().lower()
        if not drug_a or not drug_b:
            return "请提供两个药物名称，例如 {'drug_a': 'warfarin', 'drug_b': 'aspirin'}。"

        for row in self.rows:
            pair = {row["drug_a"].lower(), row["drug_b"].lower()}
            if {drug_a, drug_b} == pair:
                return (
                    f"相互作用等级：{row['severity']}\n"
                    f"说明：{row['description']}\n"
                    f"建议：{row['recommendation']}"
                )
        return "本地示例库未找到明确相互作用记录。结果不能替代医生或药师判断。"

    def _load_rows(self) -> list[dict[str, str]]:
        if not self.csv_path.exists():
            return []
        with self.csv_path.open("r", encoding="utf-8") as f:
            return list(csv.DictReader(f))
