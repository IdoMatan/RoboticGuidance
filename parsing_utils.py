from ActorCritic import Episode, Experience
from utils import *
from datetime import datetime

class EpisodeParser:
    def __init__(self,dir_path, time_interval):
        self.path = dir_path
        self.interval = time_interval
        self.episodes = []

    def load_episodes(self):
        files = os.listdir(self.path)
        # print(files)
        for file in files:
            with open(os.path.join(self.path, file)) as log_file:
                uuid = int(str(file).split("_")[1])
                episode = Episode(uuid, None)
                actions, states, next_states, traj_car, traj_drone = [], [], [], [], []
                for line in log_file:
                    line = json.loads(line)['message']
                    if type(line['state']) == list:  # not in ['START', 'GAME_END', 'target_reached']:
                        # episode.add_experience(Experience(line['state'], line['action'], line['reward'], line['next_state'], line['value'], None, uuid))
                        episode.add_experience(Experience(line['state'], line['action'], line['reward'],
                                                          line['next_state'], 0, None, uuid, line.get('drone_pose'), line.get('car_pose')), log=False)
                        actions.append(line['action'])
                        states.append(line['state'])
                        next_states.append(line['next_state'])
                        traj_drone.append(line.get('drone_pose'))
                        traj_car.append(line.get('car_pose'))

                self.episodes.append({'episode_obj': episode, 'actions': actions, 'states': states,
                                      'next_states': next_states,
                                      'traj_drone': traj_drone,
                                      'traj_car': traj_car})

        def get_uuid(episode):
            return episode['episode_obj'].uuid

        self.episodes.sort(key=get_uuid)
        for episode in self.episodes:
            print(str(datetime.utcfromtimestamp(episode["episode_obj"].uuid + 3*60*60)))

    def plot_episode(self,episode=0):

        fig, ax = plt.subplots(4)
        fig.suptitle(f'Summary: episode {str(datetime.utcfromtimestamp(self.episodes[episode]["episode_obj"].uuid + 3*60*60))}')

        ax[0].plot(self.episodes[episode]['episode_obj'].rewards, '-r', label='Rewards')
        ax[0].set_title('Rewards')
        ax[1].plot(self.episodes[episode]['actions'],'-b', label='Action')
        ax[1].set_title('Actions')
        ax[2].plot(self.episodes[episode]['states'][0], label='Relative dist')
        ax[2].plot(self.episodes[episode]['states'][1], label='Relative velocity')
        ax[2].plot(self.episodes[episode]['states'][2], label='Relative Heading')
        ax[2].set_title('State')
        ax[2].legend()
        ax[3].plot(self.episodes[episode]['states'][4], label='Drone Speed')
        ax[3].set_title('Drone Speed')
        plt.legend()
        # plt.show()

    def plot_training(self):
        avg_reward = []
        final_reward = []
        sum_reward = []

        for episode in self.episodes:
            avg_reward.append(np.mean(episode['episode_obj'].rewards))
            final_reward.append(episode['episode_obj'].rewards[-1])
            sum_reward.append(sum(episode['episode_obj'].rewards))

        fig, ax = plt.subplots(3)
        fig.suptitle('Training summary')

        ax[0].plot(avg_reward)
        ax[0].set_title('Avg reward')
        ax[1].plot(final_reward)
        ax[1].set_title('final reward')
        ax[2].plot(sum_reward)
        ax[2].set_title('sum_reward')
