import turtle
import math
import random

screen = turtle.Screen()
screen.bgcolor("black")
screen.title("Planetary Orbit Simulation")
screen.setup(width=800, height=600)
screen.tracer(0)

stars = []
for _ in range(100):
    x = random.randint(-400, 400)
    y = random.randint(-300, 300)
    brightness = random.randint(100, 255)
    stars.append([x, y, brightness])

star_drawer = turtle.Turtle()
star_drawer.hideturtle()
star_drawer.penup()
star_drawer.goto(0, -20)
star_drawer.pendown()
star_drawer.color("yellow")
star_drawer.begin_fill()
star_drawer.circle(20)
star_drawer.end_fill()

def rgb(r, g, b):
    return (r/255, g/255, b/255)

class Planet(turtle.Turtle):
    def __init__(self, a, b, radius, color, speed):
        super().__init__()
        self.a = a
        self.b = b
        self.radius = radius
        self.speed = speed
        self.color_rgb = color
        self.angle = 0
        self.penup()
        self.goto(a, 0)
        self.positions = []

    def move(self):
        self.angle += self.speed
        x = self.a * math.cos(self.angle)
        y = self.b * math.sin(self.angle)
        self.goto(x, y)
        self.positions.append([x, y])
        if len(self.positions) > 60:
            self.positions.pop(0)

drawer = turtle.Turtle()
drawer.hideturtle()
drawer.speed(0)

planets = [
    Planet(100, 80, 10, rgb(0,0,255), 0.05),
    Planet(200, 150, 15, rgb(255,0,0), 0.02),
    Planet(150, 100, 8, rgb(0,255,0), 0.03),
    Planet(250, 200, 12, rgb(128,0,128), 0.015)
]

def animate():
    drawer.clear()
    star_drawer.clear()
    for s in stars:
        s[2] += random.randint(-10, 10)
        if s[2] < 100: s[2] = 100
        if s[2] > 255: s[2] = 255
        star_drawer.penup()
        star_drawer.goto(s[0], s[1])
        star_drawer.pendown()
        star_drawer.dot(2, (s[2]/255, s[2]/255, s[2]/255))
    star_drawer.penup()
    star_drawer.goto(0, -20)
    star_drawer.pendown()
    star_drawer.color("yellow")
    star_drawer.begin_fill()
    star_drawer.circle(20)
    star_drawer.end_fill()

    for planet in planets:
        planet.move()
        for i, pos in enumerate(planet.positions):
            drawer.penup()
            drawer.goto(pos)
            drawer.pendown()
            intensity = int(255 * (i + 1) / len(planet.positions))
            r, g, b = planet.color_rgb
            drawer.dot(planet.radius, (r*intensity/255, g*intensity/255, b*intensity/255))
    screen.update()
    screen.ontimer(animate, 20)

animate()
screen.mainloop()
