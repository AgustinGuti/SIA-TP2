
ALLOWED_END_CONDITIONS = ['generations', 'structure', 'content', 'optimum']


class EndConditionConfig:
    def __init__(self, type, generations_to_check = 0, optimum_value=0, tolerance = 0, structure_tolerance = 0, structure_relevant_proportion = 0, generations=None):
        if type not in ALLOWED_END_CONDITIONS:
            raise ValueError(f"Invalid type. Valid types are: {ALLOWED_END_CONDITIONS}")
        self.type = type
        self.tolerance = tolerance
        self.generations_to_check = generations_to_check
        self.generations = generations
        self.optimum_value = optimum_value
        self.generations_with_same_best = 0
        self.current_best = None
        self.structure_tolerance = structure_tolerance
        self.structure_relevant_proportion = structure_relevant_proportion
        self.last_population = None
        self.same_structure_generations = 0

    def __str__(self):
        return f"EndConditionConfig(type={self.type}, generations_to_check={self.generations_to_check}, generations={self.generations}, optimum_value={self.optimum_value}, tolerance={self.tolerance})"

    def __repr__(self):
        return str(self)

def should_end(current_generation, population, current_best, config: EndConditionConfig):
    return END_CONDITIONS[config.type](current_generation, population, current_best, config)

def _generations_end_condition(current_generation, population, current_best, config: EndConditionConfig):
    return current_generation >= config.generations

def _structure_end_condition(current_generation, population, current_best, config: EndConditionConfig):
    if not config.last_population:
        config.last_population = population
        return False

    equal_counter = sum(
        any(config.last_population[i].variables.equals(character.variables, config.structure_tolerance) 
            for i in range(len(config.last_population))) 
        for character in population
    )

    config.last_population = population

    if equal_counter >= config.structure_relevant_proportion * len(population):
        config.same_structure_generations += 1
        return config.same_structure_generations >= config.generations_to_check
    else:
        config.same_structure_generations = 0

    return False

def _content_end_condition(current_generation, population, current_best, config: EndConditionConfig):
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
    