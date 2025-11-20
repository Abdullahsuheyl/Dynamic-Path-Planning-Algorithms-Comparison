import pygame
import heapq
import random
import math

WIDTH, HEIGHT = 1400, 800
GRID_SIZE = 20
ROWS = 30
COLS = 30
GAP = 50
FPS = 7 
BLACK = (20, 20, 20)
WHITE = (220, 220, 220)
WALL_COLOR = (80, 80, 80)
START_COLOR = (0, 255, 0)
END_COLOR = (255, 0, 0)
PATH_COLOR = (255, 215, 0)
OPEN_SET_COLOR_A = (0, 255, 0, 100)    
CLOSED_SET_COLOR_A = (0, 150, 0, 50)   
OPEN_SET_COLOR_D = (0, 100, 255, 100)  
CLOSED_SET_COLOR_D = (0, 0, 150, 50)   

pygame.init()
FONT = pygame.font.SysFont('Arial', 16)
TITLE_FONT = pygame.font.SysFont('Arial', 24, bold=True)
CMD_FONT = pygame.font.SysFont('Arial', 26, bold=True)


def heuristic(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1]) * 1.001

def get_neighbors(node, grid):
    x, y = node
    neighbors = []
    for dx, dy in [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)]:
        nx, ny = x + dx, y + dy
        if 0 <= nx < COLS and 0 <= ny < ROWS:
            if grid[ny][nx] == 0: 
                neighbors.append((nx, ny))
    return neighbors

def generate_maze(rows, cols, density=0.25):
    grid = [[0 for _ in range(cols)] for _ in range(rows)]
    for r in range(rows):
        for c in range(cols):
            if random.random() < density:
                grid[r][c] = 1
    grid[rows//2][2] = 0
    grid[rows//2][cols-3] = 0
    return grid

# --- A* ALGORİTMASI ---
class AStarStepper:
    def __init__(self, grid, start, end):
        self.grid = grid
        self.start = start
        self.end = end
        self.total_scanned = 0
        self.reset_search()

    def reset_search(self):
        self.open_set = []
        heapq.heappush(self.open_set, (0, self.start))
        self.came_from = {}
        self.g_score = {self.start: 0}
        self.closed_set = set()
        self.finished = False
        self.path = []

    def step(self):
        if not self.open_set:
            self.finished = True
            return

        for _ in range(20):
            if not self.open_set: break
            
            current = heapq.heappop(self.open_set)[1]
            
            if current in self.closed_set: continue
            self.closed_set.add(current)
            self.total_scanned += 1

            if current == self.end:
                self.reconstruct_path()
                self.finished = True
                return

            for neighbor in get_neighbors(current, self.grid):
                dist = math.hypot(neighbor[0]-current[0], neighbor[1]-current[1])
                tentative_g = self.g_score[current] + dist
                
                if tentative_g < self.g_score.get(neighbor, float('inf')):
                    self.came_from[neighbor] = current
                    self.g_score[neighbor] = tentative_g
                    f = tentative_g + heuristic(neighbor, self.end)
                    heapq.heappush(self.open_set, (f, neighbor))

    def reconstruct_path(self):
        curr = self.end
        path = []
        while curr in self.came_from:
            path.append(curr)
            curr = self.came_from[curr]
        path.append(self.start)
        self.path = path[::-1]

# --- D* LITE ---
class DStarStepper:
    def __init__(self, grid, start, end):
        self.grid = grid
        self.start = start
        self.end = end
        self.total_scanned = 0
        
        self.g = {}
        self.rhs = {}
        self.U = [] 
        self.km = 0
        
        self.processed_nodes = set()
        self.path = []
        self.finished = False
        
        for r in range(ROWS):
            for c in range(COLS):
                self.g[(c,r)] = float('inf')
                self.rhs[(c,r)] = float('inf')
        
        self.rhs[self.end] = 0
        heapq.heappush(self.U, (self.calc_key(self.end), self.end))

    def calc_key(self, u):
        val = min(self.g.get(u, float('inf')), self.rhs.get(u, float('inf')))
        return (val + heuristic(self.start, u) + self.km, val)

    def update_vertex(self, u):
        if u != self.end:
            min_rhs = float('inf')
            for nbr in get_neighbors(u, self.grid):
                dist = math.hypot(nbr[0]-u[0], nbr[1]-u[1])
                curr = self.g.get(nbr, float('inf')) + dist
                if curr < min_rhs: min_rhs = curr
            self.rhs[u] = min_rhs
        
        in_open = False
        for i, item in enumerate(self.U):
            if item[1] == u:
                self.U[i] = (self.calc_key(u), u)
                heapq.heapify(self.U)
                in_open = True
                break
        if not in_open and self.g.get(u) != self.rhs.get(u):
            heapq.heappush(self.U, (self.calc_key(u), u))

    def step(self):
        if not self.U:
            self.extract_path()
            self.finished = True
            return

        for _ in range(40): 
            if not self.U: break
            
            top_key = self.U[0][0]
            start_key = self.calc_key(self.start)
            
            if self.rhs[self.start] == self.g[self.start] and top_key >= start_key:
                valid_path = self.extract_path()
                if valid_path: 
                    self.finished = True
                    return
                else:
                    pass

            k_old, u = heapq.heappop(self.U)
            
            if k_old < self.calc_key(u): continue
            
            self.processed_nodes.add(u)
            self.total_scanned += 1
            
            if self.g.get(u, float('inf')) > self.rhs.get(u, float('inf')):
                self.g[u] = self.rhs[u]
                for nbr in get_neighbors(u, self.grid): self.update_vertex(nbr)
            else:
                self.g[u] = float('inf')
                self.update_vertex(u)
                for nbr in get_neighbors(u, self.grid): self.update_vertex(nbr)

    def on_obstacle_added(self, pos):
        self.km += heuristic(self.end, self.start)
        self.update_vertex(pos)
        for nbr in get_neighbors(pos, self.grid):
            self.update_vertex(nbr)
        self.finished = False

    def extract_path(self):
        temp_path = []
        if self.g.get(self.start) == float('inf'): 
            self.path = []
            return False 
        
        curr = self.start
        temp_path.append(curr)
        
        limit = 0
        while curr != self.end:
            limit += 1
            if limit > ROWS * COLS: 
                return False 
            
            min_val = float('inf')
            best = None
            
            for nbr in get_neighbors(curr, self.grid):
                dist = math.hypot(nbr[0]-curr[0], nbr[1]-curr[1])
                val = self.g.get(nbr, float('inf')) + dist
                if val < min_val:
                    min_val = val
                    best = nbr
            
            if best and min_val < float('inf'):
                curr = best
                temp_path.append(curr)
            else:
                return False 
        
        self.path = temp_path
        return True


def draw_panel(screen, offset_x, title, grid, open_set_vis, closed_set_vis, path, color_scheme, metric_scan):
    panel_y = 80 
    pygame.draw.rect(screen, (30, 30, 30), (offset_x-5, panel_y-5, COLS*GRID_SIZE+10, ROWS*GRID_SIZE+10))
    screen.blit(TITLE_FONT.render(title, True, WHITE), (offset_x, panel_y - 35))


    s = pygame.Surface((GRID_SIZE, GRID_SIZE))
    s.set_alpha(100)
    s.fill(color_scheme['closed'])
    for node in closed_set_vis:
        screen.blit(s, (offset_x + node[0]*GRID_SIZE, panel_y + node[1]*GRID_SIZE))

    s.set_alpha(150)
    s.fill(color_scheme['open'])
    for item in open_set_vis:
        node = item[1] 
        screen.blit(s, (offset_x + node[0]*GRID_SIZE, panel_y + node[1]*GRID_SIZE))

    for r in range(ROWS):
        for c in range(COLS):
            rect = (offset_x + c*GRID_SIZE, panel_y + r*GRID_SIZE, GRID_SIZE, GRID_SIZE)
            if grid[r][c] == 1:
                pygame.draw.rect(screen, WALL_COLOR, rect)
            pygame.draw.rect(screen, (50, 50, 50), rect, 1)

    if path and len(path) > 1:
        points = []
        for p in path:
            cx = offset_x + p[0]*GRID_SIZE + GRID_SIZE//2
            cy = panel_y + p[1]*GRID_SIZE + GRID_SIZE//2
            points.append((cx, cy))
        
        pygame.draw.lines(screen, PATH_COLOR, False, points, 5)
        for p in points: pygame.draw.circle(screen, PATH_COLOR, p, 3)


    metric_y = panel_y + ROWS*GRID_SIZE + 15
    
    pygame.draw.rect(screen, (80,80,80), (offset_x, metric_y, 200, 8))
    ratio = min(1.0, metric_scan/1500) 
    col = (0,255,0) if metric_scan<200 else (255,165,0) if metric_scan<800 else (255,0,0)
    pygame.draw.rect(screen, col, (offset_x, metric_y, 200*ratio, 8))
    screen.blit(FONT.render(f"Hesaplanan Toplam Kare: {metric_scan}", True, (200,200,200)), (offset_x + 210, metric_y - 5))

def reset_simulation():
    grid = generate_maze(ROWS, COLS)
    grid_a = [row[:] for row in grid]
    grid_d = [row[:] for row in grid]
    start = (2, ROWS//2)
    end = (COLS-3, ROWS//2)
    astar = AStarStepper(grid_a, start, end)
    dstar = DStarStepper(grid_d, start, end)
    return grid_a, grid_d, start, end, astar, dstar

def main():
    screen = pygame.display.set_mode((WIDTH, HEIGHT))
    pygame.display.set_caption("A* vs D* Lite Comparison")
    clock = pygame.time.Clock()

    grid_a, grid_d, start, end, astar, dstar = reset_simulation()
    obstacle_pos = None
    phase = 0

    run = True
    while run:
        screen.fill(BLACK)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT: run = False
            
            if event.type == pygame.MOUSEBUTTONDOWN or (event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE):
                if phase == 0: phase = 1
                elif phase == 2: 
                    if astar.path:
                        mid_idx = len(astar.path) // 2
                        obstacle_pos = astar.path[mid_idx]
                        grid_a[obstacle_pos[1]][obstacle_pos[0]] = 1
                        grid_d[obstacle_pos[1]][obstacle_pos[0]] = 1
                        astar.reset_search() 
                        dstar.on_obstacle_added(obstacle_pos)
                        phase = 3
                elif phase == 3: phase = 4
                elif phase == 5: 
                    grid_a, grid_d, start, end, astar, dstar = reset_simulation()
                    obstacle_pos = None
                    phase = 0

        # --- LOGIC ---
        info_text = ""
        info_color = WHITE
        
        if phase == 0:
            info_text = "BAŞLATMAK İÇİN TIKLAYIN"
            info_color = (100, 255, 100)
        elif phase == 1: 
            info_text = "İlk Tarama Yapılıyor..."
            astar.step(); dstar.step()
            if astar.finished and dstar.finished: phase = 2
        elif phase == 2: 
            info_text = "Yol Bulundu. Engel Eklemek İçin TIKLAYIN"
            info_color = PATH_COLOR
        elif phase == 3: 
            info_text = "Engel Eklendi. Tepkiyi Görmek İçin TIKLAYIN"
            info_color = (255, 100, 100)
        elif phase == 4: 
            info_text = "Rota Onarılıyor..."
            astar.step(); dstar.step()
            if astar.finished and dstar.finished: phase = 5
        elif phase == 5:
            info_text = "Bitti. Sıfırlamak İçin TIKLAYIN"
            info_color = (100, 200, 255)

        # --- ÇİZİM ---
        pygame.draw.rect(screen, (40, 40, 40), (0, 0, WIDTH, 60))
        txt_surf = CMD_FONT.render(info_text, True, info_color)
        screen.blit(txt_surf, (WIDTH//2 - txt_surf.get_width()//2, 15))

        draw_panel(screen, 50, "A* (Statik Planlama)", grid_a, 
                   astar.open_set, astar.closed_set, astar.path,
                   {'open': (0, 255, 0), 'closed': (0, 100, 0)}, astar.total_scanned)

        off_r = 50 + COLS*GRID_SIZE + GAP
        draw_panel(screen, off_r, "D* Lite (Dinamik Planlama)", grid_d, 
                   dstar.U, dstar.processed_nodes, dstar.path,
                   {'open': (0, 100, 255), 'closed': (0, 0, 150)}, dstar.total_scanned)

        for off in [50, off_r]:
            py = 80
            pygame.draw.rect(screen, START_COLOR, (off+start[0]*GRID_SIZE+4, py+start[1]*GRID_SIZE+4, GRID_SIZE-8, GRID_SIZE-8))
            pygame.draw.rect(screen, END_COLOR, (off+end[0]*GRID_SIZE+4, py+end[1]*GRID_SIZE+4, GRID_SIZE-8, GRID_SIZE-8))
            if obstacle_pos:
                 import time
                 if int(time.time()*5)%2 == 0:
                     pygame.draw.rect(screen, (255, 0, 255), (off+obstacle_pos[0]*GRID_SIZE, py+obstacle_pos[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE))
                     pygame.draw.rect(screen, WHITE, (off+obstacle_pos[0]*GRID_SIZE, py+obstacle_pos[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE), 2)
                 else:
                     pygame.draw.rect(screen, (200, 0, 200), (off+obstacle_pos[0]*GRID_SIZE, py+obstacle_pos[1]*GRID_SIZE, GRID_SIZE, GRID_SIZE))

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()

if __name__ == "__main__":
    main()