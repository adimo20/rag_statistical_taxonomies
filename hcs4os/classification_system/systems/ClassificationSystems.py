from ..BaseClassificationSystem import ClassificationSystem
from ..registry import register
import re

@register("SEA")
class ClassificationSystemSEA(ClassificationSystem):
    
    def get_root_categories(self) -> list[dict]:
        return [
            {"code":c.code, "description":c.description} for c in self.codes if len(c.code) == 2
        ]
    
    def get_prefixes(self, code: str) -> list[str]:
        
        code = code.replace(" ", "")
        prefixes = []
        for i in range(2,len(code)+1):
            if i <= 4:
                prefixes.append(code[:i])
            else: 
                prefixes.append(code[:4] + " " + code[4:i])
        return prefixes
    
