from gym.envs.registration import register

register(
	id='query-v0',
	entry_point='gym_query.envs:QueryEnv',
)