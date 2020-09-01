from ActorCritic import Episode, Experience
from utils import *
from datetime import datetime
from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt
import numpy as np


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
                try:
                    uuid = int(str(file).split("_")[1])
                except ValueError:
                    continue

                episode = Episode(uuid, None)
                actions, states, next_states, traj_car, traj_drone = [], [], [], [], []
                losses = None
                valid_episode = False
                for line in log_file:
                    line = json.loads(line)['message']
                    if type(line.get('state')) == list:  # not in ['START', 'GAME_END', 'target_reached']:
                        # episode.add_experience(Experience(line['state'], line['action'], line['reward'], line['next_state'], line['value'], None, uuid))
                        episode.add_experience(Experience(line['state'], line['action'], line['reward'],
                                                          line['next_state'], 0, None, uuid, line.get('drone_pose'), line.get('car_pose')), log=False)
                        actions.append(line['action'])
                        states.append(line['state'])
                        next_states.append(line['next_state'])
                        traj_drone.append(line.get('drone_pose'))
                        traj_car.append(line.get('car_pose'))

                    else:
                        if line.get('actor_loss') is not None:
                            losses = line
                            valid_episode = True


                if len(episode) > 4 and valid_episode:    # skip episodes without any experiences
                    self.episodes.append({'episode_obj': episode, 'actions': actions, 'states': states,
                                          'next_states': next_states,
                                          'traj_drone': traj_drone,
                                          'traj_car': traj_car,
                                          'losses': losses})

        def get_uuid(episode):
            return episode['episode_obj'].uuid

        self.episodes.sort(key=get_uuid)

        print(f'Total of: {len(self.episodes)}/{len(files)} valid episodes ')

        # for episode in self.episodes:
        #     print(str(datetime.utcfromtimestamp(episode["episode_obj"].uuid + 3*60*60)))

    def plot_episode(self, episode=0):

        fig, ax = plt.subplots(4)
        fig.suptitle(f'Summary: episode {str(datetime.utcfromtimestamp(self.episodes[episode]["episode_obj"].uuid + 3*60*60))}')

        dt = np.arange(len(self.episodes[episode]['actions']))

        ax[0].plot(self.episodes[episode]['episode_obj'].rewards, '-r', label='Rewards')
        ax[0].set_title('Rewards')
        ax[1].plot(dt, np.array(self.episodes[episode]['actions']),'-b', label='Action')
        ax[1].plot(dt, [row[0] for row in self.episodes[episode]['states']], label='Relative dist')
        ax[1].plot(dt, [row[1] for row in self.episodes[episode]['states']], label='Relative velocity')
        ax[1].plot(dt, [row[2]*(np.pi/180) for row in self.episodes[episode]['states']], label='Relative Heading [Rad]')
        ax[1].set_title('State-Action')
        ax[1].legend()
        ax[2].plot(dt, [row[3] for row in self.episodes[episode]['states']], label='Drone Speed')
        ax[2].set_title('Drone Speed')
        drone_traj = np.array(self.episodes[episode]['traj_drone'])
        car_traj = np.array(self.episodes[episode]['traj_car'])
        ax[3].plot(drone_traj[:,1], drone_traj[:,0], label='Drone Trajectory')
        ax[3].plot(car_traj[:,1], car_traj[:,0], label='Car Trajectory')
        ax[3].set_title('Trajectories')
        ax[3].legend()
        # ax[4].set_aspect('equal')

        plt.legend()
        # plt.show()

    def plot_training(self):
        avg_reward = []
        final_reward = []
        sum_reward = []
        ac_loss = []
        actor_loss = []
        critic_loss = []
        avg_drone_speed = []
        avg_car_speed = []

        for episode in self.episodes:
            avg_reward.append(np.mean(episode['episode_obj'].rewards))
            final_reward.append(episode['episode_obj'].rewards[-1])
            sum_reward.append(sum(episode['episode_obj'].rewards))
            avg_drone_speed.append(np.mean([row[3] for row in episode['states']]))
            avg_car_speed.append(np.mean([abs(row[3]-row[1]) for row in episode['states']]))

            if episode['losses']:
                ac_loss.append(episode.get('losses').get('ac_loss'))
                actor_loss.append(episode.get('losses').get('actor_loss'))
                critic_loss.append(episode.get('losses').get('critic_loss'))

        fig, ax = plt.subplots(5)
        fig.suptitle('Training summary')

        ax[0].plot(avg_reward)
        ax[0].set_title('Avg reward')
        ax[1].plot(final_reward)
        ax[1].set_title('final reward')
        ax[2].plot(sum_reward)
        ax[2].set_title('sum_reward')
        ax[3].plot(ac_loss, label='AC-Loss')
        ax[3].plot(actor_loss, label='Actor loss')
        ax[3].plot(critic_loss, label='Critic loss')
        ax[3].set_title('Losses')
        ax[3].legend()
        ax[4].plot(avg_drone_speed, label='Drone')
        ax[4].plot(avg_car_speed, label='Car')
        ax[4].set_title('Average Speed')
        ax[4].legend()

    def create_gif(self, episode, name='test'):
        print('Creating GIF')
        fig, ax = plt.subplots(figsize=(10, 5))

        def update(i):
            ax.plot(drone_traj[:i,1], drone_traj[:i,0], '-r', label='Drone')
            ax.plot(car_traj[:i,1], car_traj[:i,0], '-b', label='Car')
            ax.set_title(f'Time {i}', fontsize=20)
            # ax.legend()
            ax.set_axis_off()

        drone_traj = np.array(self.episodes[episode]['traj_drone'])
        car_traj = np.array(self.episodes[episode]['traj_car'])
        anim = FuncAnimation(fig, update, frames=np.arange(0, len(drone_traj)), interval=200)
        anim.save(name+'.gif', dpi=80, writer='imagemagick')
        plt.close()
