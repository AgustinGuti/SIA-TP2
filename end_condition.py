
# TODO add 'structure'
ALLOWED_END_CONDITIONS = ['generations', 'content', 'optimum']


class EndConditionConfig:
    def __init__(self, type, generations_to_check = 0, optimum_value=0, tolerance = 0, generations=None):
        if type not in ALLOWED_END_CONDITIONS:
            raise ValueError(f"Invalid type. Valid types are: {ALLOWED_END_CONDITIONS}")
        self.type = type
        self.tolerance = tolerance
        self.generations_to_check = generations_to_check
        self.generations = generations
        self.optimum_value = optimum_value
        self.generations_with_same_best = 0
        self.current_best = None

    def __str__(self):
        return f"EndConditionConfig(type={self.type}, generations_to_check={self.generations_to_check}, generations={self.generations}, optimum_value={self.optimum_value}, tolerance={self.tolerance})"

    def __repr__(self):
        return str(self)

def should_end(current_generation, current_best, config: EndConditionConfig):
    return END_CONDITIONS[config.type](current_generation, current_best, config)

def _generations_end_condition(current_generation, current_best, config: EndConditionConfig):
    return current_generation >= config.generations

def _structure_end_condition(current_generation, current_best, config: EndConditionConfig):
    # TODO implement struct end condition
    return False

def _content_end_condition(current_generation, current_best, config: EndConditionConfig):
    if config.current_best and current_best.performance <= config.current_best.performance:
        config.generations_with_same_best += 1
    else:
        config.generations_with_same_best = 0
        config.current_best = current_best
    return config.generations_with_same_best >= config.generations_to_check
    
def _optimum_end_condition(current_generation, current_best, config: EndConditionConfig):
    return abs(current_best.performance - config.optimum_value) < config.tolerance
    

END_CONDITIONS = {
    'generations': _generations_end_condition,
    'structure': _structure_end_condition,
    'content': _content_end_condition,
    'optimum': _optimum_end_condition
}
    