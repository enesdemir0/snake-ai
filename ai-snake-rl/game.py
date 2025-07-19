import random
from agent import Agent
import math
import numpy as np
import matplotlib.pyplot as plt
import os
import time
import pygame
import datetime

pygame.init()

# Game settings
WIDTH, HEIGHT = 400, 400
BLOCK_SIZE = 20
FPS = 30

UP = (0, -1)
DOWN = (0, 1)
LEFT = (-1, 0)
RIGHT = (1, 0)
DIRECTION_ORDER = [UP, RIGHT, DOWN, LEFT]


class Game:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.block_size = BLOCK_SIZE
        self.reset()
        self.prev_food = None

    def reset(self):
        self.snake = [(self.width // 2, self.height // 2)]
        self.direction = RIGHT
        self.spawn_food()
        self.game_over = False
        self.ate_food = False
        self.previous_distance = self.distance_to_food()

    def spawn_food(self):
        self.prev_food = self.food if hasattr(self, 'food') else None
        grid_width = self.width // self.block_size
        grid_height = self.height // self.block_size
        while True:
            x = random.randint(0, grid_width - 1) * self.block_size
            y = random.randint(0, grid_height - 1) * self.block_size
            if (x, y) not in self.snake:
                self.food = (x, y)
                break

    def move_snake(self, action):
        idx = DIRECTION_ORDER.index(self.direction)
        if action == 0:
            idx = (idx - 1) % 4
        elif action == 2:
            idx = (idx + 1) % 4
        self.direction = DIRECTION_ORDER[idx]

        head_x, head_y = self.snake[0]
        delta_x, delta_y = self.direction
        new_head = (head_x + delta_x * self.block_size, head_y + delta_y * self.block_size)
        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.ate_food = True
            self.spawn_food()
        else:
            self.ate_food = False
            self.snake.pop()

    def update(self):
        head = self.snake[0]
        x, y = head
        if x < 0 or x >= self.width or y < 0 or y >= self.height:
            self.game_over = True
        if head in self.snake[1:]:
            self.game_over = True

    def get_state(self, last_actions=None):
        head = self.snake[0]
        x, y = head

        point_l = (x - self.block_size, y)
        point_r = (x + self.block_size, y)
        point_u = (x, y - self.block_size)
        point_d = (x, y + self.block_size)

        dir_l = self.direction == LEFT
        dir_r = self.direction == RIGHT
        dir_u = self.direction == UP
        dir_d = self.direction == DOWN

        def danger(point):
            px, py = point
            if px < 0 or px >= self.width or py < 0 or py >= self.height:
                return True
            if point in self.snake[1:]:
                return True
            return False

        def distance_to_wall(dx, dy):
            d = 0
            px, py = x, y
            while 0 <= px < self.width and 0 <= py < self.height and (px, py) not in self.snake:
                d += 1
                px += dx * self.block_size
                py += dy * self.block_size
            return d / (self.width // self.block_size)

        food_dx = self.food[0] - x
        food_dy = self.food[1] - y

        euclidean = ((food_dx) ** 2 + (food_dy) ** 2) ** 0.5 / ((self.width ** 2 + self.height ** 2) ** 0.5)
        manhattan = (abs(food_dx) + abs(food_dy)) / (self.width + self.height)

        angle_to_food = np.arctan2(food_dy, food_dx) / np.pi  # normalize -1 to 1

        # Base 19 features
        state = [
            int(danger(point_r)),  # danger straight (relative to current direction RIGHT)
            int(danger(point_d)),  # danger right
            int(danger(point_u)),  # danger left
            int(dir_l),
            int(dir_r),
            int(dir_u),
            int(dir_d),
            int(self.food[0] < x),
            int(self.food[0] > x),
            int(self.food[1] < y),
            int(self.food[1] > y),
            euclidean,
            manhattan,
            angle_to_food,
            distance_to_wall(1, 0),  # distance to wall right
            distance_to_wall(-1, 0),  # left
            distance_to_wall(0, -1),  # up
            distance_to_wall(0, 1),  # down
        ]

        # Is food in same direction as current movement
        food_direction = 0
        if dir_r and self.food[0] > x:
            food_direction = 1
        elif dir_l and self.food[0] < x:
            food_direction = 1
        elif dir_u and self.food[1] < y:
            food_direction = 1
        elif dir_d and self.food[1] > y:
            food_direction = 1

        state.append(int(food_direction))  # 19 features total

        # === MEMORY/HISTORY FEATURES ===

        # 1) Distance to tail (normalized)
        tail = self.snake[-1]
        dist_to_tail = ((tail[0] - x) ** 2 + (tail[1] - y) ** 2) ** 0.5 / ((self.width ** 2 + self.height ** 2) ** 0.5)
        state.append(dist_to_tail)

        # 2) Loop detection - Is head close to any body part except neck and tail?
        loop = 0
        for part in self.snake[2:-1]:
            if abs(part[0] - x) <= self.block_size and abs(part[1] - y) <= self.block_size:
                loop = 1
                break
        state.append(loop)

        # 3) Last 3 actions one-hot encoded (if not given, assume all straight = 1)
        if last_actions is None:
            last_actions = [1, 1, 1]
        for action in last_actions[-3:]:
            state.extend([
                1 if action == 0 else 0,
                1 if action == 1 else 0,
                1 if action == 2 else 0
            ])

        # 4) Relative tail position
        state.append(int(tail[0] < x))
        state.append(int(tail[0] > x))
        state.append(int(tail[1] < y))
        state.append(int(tail[1] > y))

        # Now total ~34 features

        return state

    def distance_to_food(self):
        head = self.snake[0]
        return math.sqrt((head[0] - self.food[0]) ** 2 + (head[1] - self.food[1]) ** 2)


# Plot training scores (used in main, not evaluation)
def plot_training_progress(episodes, scores):
    plt.figure(figsize=(10, 5))
    plt.plot(episodes, scores, '-b')
    plt.xlabel('Episode')
    plt.ylabel('Score (Food eaten)')
    plt.title('Snake AI Training Progress')
    plt.grid(True)
    plt.show()


def evaluate_model_with_ui():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sanek AI Snake - Evaluation Mode")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)

    game = Game()
    agent = Agent(state_size=34)
    agent.load("models/sanek_model_45_20250718_201042")
    agent.epsilon = 0

    evaluating = False
    total_score = 0
    num_games = 0
    current_score = 0

    while True:
        screen.fill((0, 0, 0))

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE and not evaluating:
                    game.reset()
                    evaluating = True
                    current_score = 0

        if evaluating and not game.game_over:
            state = game.get_state()
            action = agent.get_action(state)
            game.move_snake(action)
            game.update()

        if evaluating and game.game_over:
            evaluating = False
            score = len(game.snake) - 1
            total_score += score
            num_games += 1
            current_score = score

        # Draw snake and food
        for block in game.snake:
            pygame.draw.rect(screen, (0, 255, 0), (*block, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(screen, (255, 0, 0), (*game.food, BLOCK_SIZE, BLOCK_SIZE))

        # Display score info
        screen.blit(font.render("Press SPACE to evaluate a game", True, (255, 255, 255)), (10, 10))
        screen.blit(font.render(f"Last Score: {current_score}", True, (255, 255, 0)), (10, 40))
        screen.blit(font.render(f"Games Played: {num_games}", True, (200, 200, 200)), (10, 70))
        screen.blit(
            font.render(f"Average Score: {total_score / num_games:.2f}" if num_games else "", True, (200, 200, 200)),
            (10, 100))

        pygame.display.flip()
        clock.tick(FPS)


def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("Sanek AI Snake")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    game = Game()
    agent = Agent(state_size=34)

    # Create models folder if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Load best model (if exists)
    model_loaded = False
    if os.path.exists("models/sanek_model_45_20250718_201042_agent.npz") and os.path.exists(
            "models/sanek_model_45_20250718_201042_nn.npz"):
        agent.load("models/sanek_model_45_20250718_201042")
        model_loaded = True
        print("Loaded best model from 'models/sanek_model_best'")
    else:
        print(" No previous best model found, training from scratch")

    #  Load best score
    best_score = 0
    if os.path.exists("best_score.txt"):
        with open("best_score.txt", "r") as f:
            try:
                best_score = int(f.read().strip())
                print(f"Loaded best score: {best_score}")
            except ValueError:
                print("Corrupted best_score.txt — starting from 0.")
                best_score = 0

    agent.epsilon = 0.1  # Optional: allow small exploration
    episode = 0
    scores = []
    episodes = []
    game_over_time = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                print("Saving current model before quitting...")
                agent.save("models/sanek_model_latest")
                running = False

        if not game.game_over:
            state = game.get_state()
            action = agent.get_action(state)

            previous_distance = game.previous_distance
            game.move_snake(action)
            game.update()
            new_distance = game.distance_to_food()
            game.previous_distance = new_distance

            # Reward system
            if game.game_over:
                reward = -10
                done = True
                game_over_time = time.time()
            elif game.ate_food:
                reward = 10
                done = False
            elif new_distance < previous_distance:
                reward = 1
                done = False
            else:
                reward = -1
                done = False

            next_state = game.get_state()
            agent.train(state, action, reward, next_state, done)

        else:
            if game_over_time and (time.time() - game_over_time) > 1.0:
                episode += 1
                score = len(game.snake) - 1
                scores.append(score)
                episodes.append(episode)

                print(f"Episode {episode} | Score: {score} | Epsilon: {agent.epsilon:.4f}")

                # Save new model only if score improves
                if score > best_score:
                    best_score = score
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    model_filename = f"models/sanek_model_{score}_{timestamp}"
                    agent.save(model_filename)

                    # Save best score
                    with open("best_score.txt", "w") as f:
                        f.write(str(best_score))

                    print(f" New model saved as '{model_filename}' with score {score}!")
                    print(f" 'models/sanek_model_best' is NOT overwritten — safe and sound!")

                game.reset()
                game_over_time = None

                #  Reset epsilon if too greedy
                if agent.epsilon < 0.05:
                    agent.reset_epsilon(0.1)

        #  Draw game
        screen.fill((0, 0, 0))
        for block in game.snake:
            pygame.draw.rect(screen, (0, 255, 0), (*block, BLOCK_SIZE, BLOCK_SIZE))
        pygame.draw.rect(screen, (255, 0, 0), (*game.food, BLOCK_SIZE, BLOCK_SIZE))

        if game.game_over:
            text = font.render("Game Over! Restarting...", True, (255, 255, 255))
            screen.blit(text, (10, HEIGHT // 2))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

    # Plot if training was long enough
    if len(scores) > 1:
        plot_training_progress(episodes, scores)


def evaluate_model(agent, model_prefix, episodes=1000, max_steps=1000):
    agent.load(model_prefix)
    agent.epsilon = 0  # Evaluation only
    print(f"Loaded model and epsilon={agent.epsilon:.4f}")

    scores = []

    for _ in range(episodes):
        game = Game()
        score = 0
        steps = 0

        while not game.game_over and steps < max_steps:
            state = game.get_state()
            action = agent.get_action(state)
            game.move_snake(action)
            game.update()
            steps += 1

        score = len(game.snake) - 1
        scores.append(score)

    avg_score = np.mean(scores)
    min_score = np.min(scores)
    max_score = np.max(scores)

    return {
        "model": model_prefix,
        "average": avg_score,
        "min": min_score,
        "max": max_score
    }


def evaluate_all_models():
    models_dir = "models"
    print(" Scanning for model pairs in 'models/' folder...")

    # Find all model prefixes with both _agent.npz and _nn.npz
    model_prefixes = set()
    for filename in os.listdir(models_dir):
        if filename.startswith("sanek_model_") and filename.endswith("_agent.npz"):
            base = filename.replace("_agent.npz", "")
            if os.path.exists(os.path.join(models_dir, base + "_nn.npz")):
                model_prefixes.add(base)

    if not model_prefixes:
        print(" No valid model pairs found.")
        return

    results = []

    for model_name in sorted(model_prefixes):
        model_path = os.path.join(models_dir, model_name)
        print(f"Evaluating {model_path}...")

        agent = Agent(state_size=34)
        result = evaluate_model(agent, model_path, episodes=50, max_steps=1000)
        results.append(result)

    #  Sort results by average score descending
    results.sort(key=lambda x: x["average"], reverse=True)

    print("\n Evaluation Results:")
    for res in results:
        print(f" {res['model']} | Avg: {res['average']:.2f} | Min: {res['min']} | Max: {res['max']}")


if __name__ == "__main__":
    # Uncomment one of these to run the desired mode:

    # To train the model interactively:
    # main()

    # To evaluate all models in the models/ directory:
    # evaluate_all_models()

    # To run the game with GUI playing using a single loaded model:
    evaluate_model_with_ui()
