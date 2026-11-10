"""使用 AST 白名单解释四则运算，不执行表达式中的 Python 代码。"""

import ast
import math


_MAX_LENGTH = 200
_MAX_NODES = 50
_MAX_ABSOLUTE_VALUE = 10**12
_ALLOWED_NODES = (
    ast.Expression,
    ast.Constant,
    ast.BinOp,
    ast.UnaryOp,
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.UAdd,
    ast.USub,
)


class _CalculationError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _error(code: str, message: str) -> dict:
    return {"ok": False, "error": {"code": code, "message": message}}


def _bounded(value: int | float) -> int | float:
    if abs(value) > _MAX_ABSOLUTE_VALUE or not math.isfinite(value):
        raise _CalculationError(
            "numeric_limit", "数字、每一步中间结果及最终结果必须有限，且绝对值不超过 1e12。"
        )
    return value


def _calculate_node(node: ast.AST) -> int | float:
    if isinstance(node, ast.Constant):
        return _bounded(node.value)
    if isinstance(node, ast.UnaryOp):
        operand = _calculate_node(node.operand)
        if isinstance(node.op, ast.UAdd):
            return _bounded(+operand)
        if isinstance(node.op, ast.USub):
            return _bounded(-operand)
    if isinstance(node, ast.BinOp):
        left = _calculate_node(node.left)
        right = _calculate_node(node.right)
        if isinstance(node.op, ast.Add):
            return _bounded(left + right)
        if isinstance(node.op, ast.Sub):
            return _bounded(left - right)
        if isinstance(node.op, ast.Mult):
            return _bounded(left * right)
        if isinstance(node.op, ast.Div):
            if right == 0:
                raise _CalculationError("division_by_zero", "除数不能为零。")
            return _bounded(left / right)
    raise _CalculationError("invalid_expression", "只允许数字、括号、一元正负号和加减乘除。")


def calculate(expression: str) -> dict:
    """安全解释数值表达式，返回原始表达式和可 JSON 序列化的数值。"""
    if not isinstance(expression, str):
        return _error("invalid_arguments", "expression 必须是字符串。")
    # 长度按原始输入计算，不能通过剔除空白绕过限制。
    if len(expression) > _MAX_LENGTH:
        return _error("expression_too_long", "表达式不能超过 200 个字符（包含空白）。")
    if not expression.strip():
        return _error("invalid_expression", "表达式不能为空。")
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        nodes = list(ast.walk(tree))
        if len(nodes) > _MAX_NODES:
            return _error("expression_too_complex", "表达式 AST 不能超过 50 个节点（包含运算符）。")
        # 先检查整棵树；bool 虽是 int 的子类，也必须明确排除。
        for node in nodes:
            if type(node) not in _ALLOWED_NODES or (
                isinstance(node, ast.Constant) and type(node.value) not in (int, float)
            ):
                return _error("invalid_expression", "只允许数字、括号、一元正负号和加减乘除。")
        value = _calculate_node(tree.body)
    except _CalculationError as exc:
        return _error(exc.code, str(exc))
    except (SyntaxError, ValueError):
        return _error("invalid_expression", "表达式格式不正确，请使用数字、括号和加减乘除。")
    except RecursionError:
        return _error("expression_too_complex", "表达式嵌套过深。")
    except OverflowError:
        return _error("numeric_limit", "计算结果超出允许范围，绝对值不能超过 1e12。")
    return {"ok": True, "expression": expression, "value": value}


