import sys, pygame
import pygame.locals

SPACING = 50
INITIAL = SPACING * 4


class GUI():
  def __init__(self, hexes=None):
    pygame.init()
    self.screen = pygame.display.set_mode((SPACING * 20, SPACING*20))
    self.screen.fill((255, 255, 255))
    self.draw_board(hexes=hexes)

  def draw_hexagon(self, topX, topY):
    coordinates = [(topX, topY), (topX-SPACING, topY+SPACING),(topX-SPACING, topY+ 2*SPACING),(topX, topY+3*SPACING),(topX+SPACING, topY+2*SPACING),(topX+SPACING, topY+SPACING)]
    pygame.draw.polygon(self.screen, 0x000000, coordinates, 2)


  def draw_row(self, amountOfHexes, startX, startY):
    for i in range(amountOfHexes):
      self.draw_hexagon(startX, startY)
      startX += 2 * SPACING

  def draw_board(self, hexes=None):
    hexes = hexes or [3, 4, 5, 4, 3]
    startXs = [INITIAL, INITIAL - SPACING, INITIAL - 2*SPACING, INITIAL-SPACING, INITIAL]
    startYs = [INITIAL, INITIAL + 2*SPACING, INITIAL + 4*SPACING, INITIAL + 6*SPACING, INITIAL + 8*SPACING]
    for i in range(len(hexes)):
      self.draw_row(hexes[i], startXs[i], startYs[i])

  def draw_roads(self, roads):
    startXs = [INITIAL - SPACING, INITIAL - SPACING, INITIAL - 2 * SPACING, INITIAL - 2 * SPACING, INITIAL - 3 * SPACING, INITIAL - 3 * SPACING, INITIAL - 3 * SPACING, INITIAL - 2 * SPACING, INITIAL - 2 * SPACING, INITIAL - SPACING, INITIAL - SPACING]
    startYs = [INITIAL + SPACING, INITIAL + SPACING, INITIAL + 3 * SPACING, INITIAL + 3 * SPACING, INITIAL + 5 * SPACING, INITIAL + 5 * SPACING, INITIAL + 6 * SPACING, INITIAL + 8 * SPACING, INITIAL + 8 * SPACING, INITIAL + 10 * SPACING, INITIAL + 10* SPACING]
    colors = {1: 0xff0000, 2: 0x0000ff}
    for row in range(len(roads)):
      X = startXs[row]
      Y = startYs[row]
      for col in range(len(roads[row])):
        if (row <= 5):
          # Horizontal roads
          if (row % 2 == 0):
            # Road going up
            if (col % 2 == 0):
              if (roads[row][col] != 0):
                pygame.draw.line(self.screen, colors[roads[row][col]], (X, Y), (X + SPACING, Y - SPACING), 7)
            # Road going down
            else:
              if (roads[row][col] != 0):
                pygame.draw.line(self.screen, colors[roads[row][col]], (X, Y - SPACING), (X + SPACING, Y), 7)
            X += SPACING
          # Vertical roads
          else:
            if (roads[row][col] != 0):
              pygame.draw.line(self.screen, colors[roads[row][col]], (X, Y), (X, Y + SPACING), 7)
            X += 2 * SPACING
        elif (row == 6):
          # Road going up
          if (col % 2 != 0):
            if (roads[row][col] != 0):
              pygame.draw.line(self.screen, colors[roads[row][col]], (X, Y + SPACING), (X + SPACING, Y), 7)
          # Road going down
          else:
            if (roads[row][col] != 0):
              pygame.draw.line(self.screen, colors[roads[row][col]], (X, Y), (X + SPACING, Y + SPACING), 7)
          X += SPACING
        else:
          # Horizontal roads
          if (row % 2 == 0):
            # Road going down
            if (col % 2 == 0):
              if (roads[row][col] != 0):
                pygame.draw.line(self.screen, colors[roads[row][col]], (X, Y), (X + SPACING, Y + SPACING), 7)
            # Road going up
            else:
              if (roads[row][col] != 0):
                pygame.draw.line(self.screen, colors[roads[row][col]], (X, Y + SPACING), (X + SPACING, Y), 7)
            X += SPACING
          # Vertical roads
          else:
            if (roads[row][col] != 0):
              pygame.draw.line(self.screen, colors[roads[row][col]], (X, Y), (X, Y - SPACING), 7)
            X += 2*SPACING

  def draw_settlements(self, settlements):
    startXs = [INITIAL - SPACING, INITIAL - 2 * SPACING, INITIAL - 3 * SPACING, INITIAL - 3 * SPACING, INITIAL - 2 * SPACING, INITIAL - 1 * SPACING]
    startYs = [INITIAL + SPACING, INITIAL + 3 * SPACING, INITIAL + 5 * SPACING, INITIAL + 6 * SPACING, INITIAL + 8 * SPACING, INITIAL + 10 * SPACING]
    colors = {1: 0xff0000, 2: 0x0000ff}
    for row in range(len(settlements)):
      X = startXs[row]
      Y = startYs[row]
      for col in range(len(settlements[row])):
        if (settlements[row][col] != 0):
          pygame.draw.circle(self.screen, colors[settlements[row][col]], (X, Y), 10)
        X += SPACING
        # Leftmost edge going down
        if (row > 2):
          if (col % 2 == 0):
            Y += SPACING
          else:
            Y -= SPACING
        # Leftmost edge going up
        else:
          if (col % 2 == 0):
            Y -= SPACING
          else:
            Y += SPACING

  def draw_text(self, player1, player2):
    font = pygame.font.Font('freesansbold.ttf', 32)
    text = font.render(player1, True, (255, 0, 0), (255, 255, 255))
    textRect = text.get_rect()
    textRect.center = (700, 100)
    self.screen.blit(text, textRect)
    text = font.render(player2, True, (0, 0, 255), (255, 255, 255))
    textRect = text.get_rect()
    textRect.center = (700, 150)
    self.screen.blit(text, textRect)

  def show(self):
    pygame.display.flip()

  def save(self, location):
    pygame.image.save(self.screen, location)

  def close(self):
    pygame.quit()