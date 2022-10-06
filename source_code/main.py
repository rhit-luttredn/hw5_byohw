import os
import math

import neat
import visualize
import pandas as pd
import numpy as np
import json

# load data
print("Loading data")
data = pd.read_csv("sudoku_cleaned.csv")
print("Preprocessing Data")
data = data.apply(lambda x: x.apply(json.loads))

# get test and train sets
FRAC_TRAIN = 0.5

data = data.sample(frac=1).reset_index(drop=True)
train = data.head(math.floor(data.shape[0] * FRAC_TRAIN))
train_x = train["quizzes"]
train_y = train["solutions"]

test = data.tail(math.ceil(data.shape[0] * FRAC_TRAIN))
test_x = test["quizzes"]
test_y = test["solutions"]


def eval_genomes(genomes, config):
    print("Eval Genomes")
    sample = train.sample(n=1)
    inputs = sample.quizzes
    outputs = sample.solutions
    for genome_id, genome in genomes:
        print(f"genome: {genome_id}")
        genome.fitness = 81
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        for xi, yi in zip(inputs, outputs):
            print((np.reshape(xi, (9,9)) + 0.5) * 9)
            while not 0 in xi:
                output = net.activate(xi)
                indices = np.flip(np.argsort(output))

                # find idx to apply
                max_idx = -1
                for idx in indices:
                    if xi[math.floor(idx/9)] == 0:
                        max_idx = idx
                        break
                
                board_idx = math.floor(max_idx / 9)
                number = max_idx % 9
                print(board_idx)
                print(number)
                xi[board_idx] = number / 9 - 0.5
                genome.fitness -= int(xi[board_idx] != yi[board_idx])
                # print(np.reshape(xi, (9,9)))


def run(config_file):
    print("Configuring Settings")
    # Load configuration.
    config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                         neat.DefaultSpeciesSet, neat.DefaultStagnation,
                         config_file)

    # Create the population, which is the top-level object for a NEAT run.
    print("Creating Population")
    p = neat.Population(config)

    # Add a stdout reporter to show progress in the terminal.
    p.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    p.add_reporter(stats)
    p.add_reporter(neat.Checkpointer(5, filename_prefix="checkpoints/neat-checkpoint-"))

    # Run for up to 300 generations.
    print("Training Network")
    winner = p.run(eval_genomes, 300)
    print("Finished Training")

    # Display the winning genome.
    print('\nBest genome:\n{!s}'.format(winner))

    # Show output of the most fit genome against training data.
    print('\nOutput:')
    winner_net = neat.nn.FeedForwardNetwork.create(winner, config)
    for xi, yi in zip(train_x, train_y):
        output = winner_net.activate(xi)
        print("input {!r}, expected output {!r}, got {!r}".format(xi, yi, output))

    node_names = {-1: 'A', -2: 'B', 0: 'A XOR B'}
    visualize.draw_net(config, winner, True, node_names=node_names)
    # visualize.draw_net(config, winner, True, node_names=node_names, prune_unused=True)
    visualize.plot_stats(stats, ylog=False, view=True)
    visualize.plot_species(stats, view=True)

    # p = neat.Checkpointer.restore_checkpoint('checkpoints/neat-checkpoint-4')
    # p.run(eval_genomes, 10)


if __name__ == '__main__':
    # Determine path to configuration file. This path manipulation is
    # here so that the script will run successfully regardless of the
    # current working directory.
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'properties.config')
    run(config_path)