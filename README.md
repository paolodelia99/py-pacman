# PyPacman

The classic game of Pacman built with Pygame.

![example](res/pacman-example.gif)

# Quick Start

Install the requirements
    
    pip install -r requirements.txt
    
Run the Game with the classic maze

    python main.py -lay classic -snd

Run the Game without music or sounds 

    python main.py -lay classic

Run the game with others option

    usage: main.py [-h] [-lay LAYOUT] [-snd] [-stt]

    Argument for the Pacman Game
    
    optional arguments:
      -h, --help            show this help message and exit
      -lay LAYOUT, --layout LAYOUT
                            Name of layout to load in the game
      -snd, --sound         Activate sounds in the game
      -stt, --state         Display the state matrix of the game
        
# Todos

- [ ] implement fruit
- [ ] flashing power pellet
- [x] state matrix in another screen
- [ ] Provide an RL Environment so an AI agent can be trained

# License

MIT

# Author

Paolo D'Elia