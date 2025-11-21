import random
import operator

class MathGenerator:
    def __init__(self):
        self.ops_map = {
            '+': operator.add,
            '-': operator.sub,
            '*': operator.mul,
            # '/': operator.truediv  # Keeping it simple for now, can add later
        }

    def get_config_for_difficulty(self, difficulty):
        """
        Returns config dict based on difficulty (1-10).
        """
        # Scale depth and max_value based on difficulty
        # Difficulty 1: depth 1, digits 1 (0-9)
        # Difficulty 5: depth 3, digits 2 (0-99)
        # Difficulty 10: depth 5, digits 3 (0-999)
        
        depth = 1 + (difficulty // 2)
        digits = 1 + (difficulty // 4)
        max_val = 10**digits
        
        return {
            'max_depth': depth,
            'max_value': max_val,
            'ops': ['+', '-'] if difficulty < 3 else ['+', '-', '*']
        }

    def generate_expression(self, difficulty=1, current_depth=0, target_depth=None, config=None):
        """
        Generates a random math expression based on difficulty.
        """
        if config is None:
            config = self.get_config_for_difficulty(difficulty)
            target_depth = config['max_depth']

        # Base case: Leaf node (number)
        # We stop if we reached max depth OR randomly if we are not forced to go deeper
        # But to ensure complexity, let's try to reach target_depth somewhat
        is_leaf = (current_depth >= target_depth) or (current_depth > 0 and random.random() < 0.2)

        if is_leaf:
            val = random.randint(1, config['max_value'])
            return str(val), val
        
        # Internal node
        op_symbol = random.choice(config['ops'])
        op_func = self.ops_map[op_symbol]
        
        left_str, left_val = self.generate_expression(difficulty, current_depth + 1, target_depth, config)
        right_str, right_val = self.generate_expression(difficulty, current_depth + 1, target_depth, config)
        
        expr_str = f"({left_str} {op_symbol} {right_str})"
        result_val = op_func(left_val, right_val)
        
        return expr_str, result_val

    def generate_batch(self, batch_size, difficulty=1):
        batch = []
        for _ in range(batch_size):
            expr, res = self.generate_expression(difficulty=difficulty)
            full_str = f"{expr}={res}"
            batch.append(full_str)
        return batch

if __name__ == "__main__":
    gen = MathGenerator()
    print("Difficulty 1:")
    for _ in range(3): print(gen.generate_expression(difficulty=1))
    print("\nDifficulty 5:")
    for _ in range(3): print(gen.generate_expression(difficulty=5))
    print("\nDifficulty 10:")
    for _ in range(3): print(gen.generate_expression(difficulty=10))
