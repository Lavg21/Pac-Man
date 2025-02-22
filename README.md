# 🎮 Pac-Man AI Project

> A project implementing search algorithms and adversarial strategies for Pac-Man.

## 📌 Overview

This project explores **search problems** and **adversarial agents** in **Pac-Man AI**, including **uninformed search**, **informed search**, and **minimax-based decision making**. The implemented solutions are designed to improve **pathfinding, food collection, and agent competition**.

## 🛠 Features

✅ **Depth-First Search (DFS) & Breadth-First Search (BFS)**  
✅ **Uniform-Cost Search (UCS) & A* Search Algorithm**  
✅ **Corners Problem Heuristic**  
✅ **Food Collection Strategies**  
✅ **Minimax & Alpha-Beta Pruning**  
✅ **Iterative Deepening Search**  

## 📜 Table of Contents

- [Implemented Algorithms](#implemented-algorithms)
- [Installation](#installation)
- [Usage](#usage)
- [Testing](#testing)
- [Technologies Used](#technologies-used)

## 🔍 Implemented Algorithms

### 1️⃣ Uninformed Search
- **Depth-First Search (DFS)**: Implements a **LIFO-based** graph traversal.
- **Breadth-First Search (BFS)**: Uses a **FIFO queue** to explore shallow nodes first.
- **Uniform-Cost Search (UCS)**: Expands nodes with **least path cost**.

### 2️⃣ Informed Search
- **A* Search**: Utilizes an **admissible and consistent heuristic**.
- **Corners Problem Heuristic**: Optimized heuristic for visiting all corners.
- **Food Collection Heuristic**: Uses **maze distance** to prioritize eating food efficiently.

### 3️⃣ Adversarial Search
- **Minimax Algorithm**: Pac-Man competes against ghosts using minimax-based **decision trees**.
- **Alpha-Beta Pruning**: Optimizes Minimax by eliminating unnecessary branches.
- **Reflex Agent**: Implements a smarter AI to maximize Pac-Man’s survival.
- **Iterative Deepening DFS**: Enhances search efficiency by incrementally increasing depth.

## 💻 Installation

### Prerequisites
- Python 3.8+
- `pygame` for visualizing Pac-Man
- `numpy` for heuristic calculations

## 🎮 Usage

1. **Run Pac-Man AI**
   ```bash
   python pacman.py -l mediumMaze -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic
   ```
2. **Test Adversarial Search (Minimax)**
   ```bash
   python pacman.py -p MinimaxAgent -l minimaxClassic -a depth=4
   ```
3. **Test Alpha-Beta Pruning**
   ```bash
   python pacman.py -p AlphaBetaAgent -a depth=3 -l smallClassic
   ```

## 🧪 Testing

### **Basic Search Algorithms**
- `python pacman.py -l tinyMaze -p SearchAgent -a fn=dfs`
- `python pacman.py -l mediumMaze -p SearchAgent -a fn=bfs`
- `python pacman.py -l bigMaze -z .5 -p SearchAgent -a fn=ucs`

### **Informed Search**
- `python pacman.py -l bigMaze -p SearchAgent -a fn=astar,heuristic=manhattanHeuristic`

### **Adversarial Search**
- `python pacman.py -p ReflexAgent -l testClassic`
- `python pacman.py -p MinimaxAgent -a depth=4`
- `python pacman.py -p AlphaBetaAgent -a depth=3`

## ⚡ Technologies Used

- **Python**: Core programming language
- **Pac-Man AI Framework**: Provided for developing search strategies
- **Algorithms**: BFS, DFS, UCS, A*, Minimax, Alpha-Beta Pruning
- **Heuristics**: Manhattan distance, maze distance
