from random import randint

class Cell:

  def __init__(self, id, current_epoch, is_active = False, x = None, y = None):
    self.id = id
    self.active = is_active
    self.x = x
    self.y = y
    self.birthdate=current_epoch

  def toggle_active(self):
    self.active = not self.active
  
  def set_location(self, x = None, y = None):
    self.x = x
    self.y = y
  
  def set_random_location(self, grid_size):
    self.x = randint(0, grid_size-1)
    self.y = randint(0, grid_size-1) 
    