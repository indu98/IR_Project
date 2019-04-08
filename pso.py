from __future__ import division
import numpy as np
import random
import math
import os
import sys

#--- COST FUNCTION ------------------------------------------------------------+

# function we are attempting to optimize (minimize)
def func1(x):
	a = 0.00001
	diff = 0
	avg = sum(x)/float(len(x))
	for i in range (0,len(x)):
		if x[i]==-1 or x[i]==0:
			t=0
		else:
			t=1
		diff = diff+t-avg
		if x[i]>0:
			a = a+1

	total = diff/a 
	return total

#--- MAIN ---------------------------------------------------------------------+

class Particle:
	def __init__(self,x0):
		self.position_i=[]          # particle position
		self.velocity_i=[]          # particle velocity
		self.pos_best_i=[]          # best position individual
		self.err_best_i=0.1          # best error individual
		self.err_i=0.1               # error individual

		for i in range(0,num_dimensions):
			self.velocity_i.append(random.uniform(-1,1))
			self.position_i.append(x0[i])

	# evaluate current fitness
	def evaluate(self,costFunc):
		self.err_i=costFunc(self.position_i)

	# check to see if the current position is an individual best
		if self.err_i > self.err_best_i or self.err_best_i==0.1:
			self.pos_best_i=list(self.position_i)
			self.err_best_i=self.err_i
                    
	# update new particle velocity
	def update_velocity(self,pos_best_g):
		w=0.5       # constant inertia weight (how much to weigh the previous velocity)
		c1=1        # cognative constant
		c2=2        # social constant
        
		for i in range(0,num_dimensions):
			r1=random.random()
			r2=random.random()
            
			vel_cognitive=c1*r1*(self.pos_best_i[i]-self.position_i[i])
			vel_social=c2*r2*(pos_best_g[i]-self.position_i[i])
			self.velocity_i[i]=w*self.velocity_i[i]+vel_cognitive+vel_social

	# update the particle position based off new velocity updates
	def update_position(self):
		for i in range(0,num_dimensions):
			s=0
			lim = 1.0/(1+ math.exp(-(self.velocity_i[i])))
			if random.uniform(0,1) < lim :
				s = 1
			if random.uniform(0,1) > s and self.position_i[i]>-1 :	
				self.position_i[i] = 0
            
		
class PSO():
	def __init__(self, costFunc, x0,num_particles, maxiter, verbose=False):
		global num_dimensions
		
		global X_PSO
		X_PSO=[]
		
		err_best_g=0.1                   # best error for group
		pos_best_g=[]                   # best position for group

		# establish the swarm
		swarm=[]
		#print num_particles
		for i in range(0,num_particles):
			num_dimensions=len(x0[i])
			#print x0[i]
			swarm.append(Particle(x0[i]))
			#print swarm[i].position_i

		# begin optimization loop
		i=0
		while i<maxiter:
			if verbose: 
				print('iter:',i, 'best solution:',err_best_g)
			# cycle through particles in swarm and evaluate fitness
			for j in range(0,num_particles):
				swarm[j].evaluate(costFunc)

			# determine if current particle is the best (globally)
				if swarm[j].err_i>err_best_g or err_best_g==0.1:
					pos_best_g=list(swarm[j].position_i)
					err_best_g=float(swarm[j].err_i)
            
			# cycle through swarm and update velocities and position
			for j in range(0,num_particles):
				swarm[j].update_velocity(pos_best_g)
				swarm[j].update_position()
				#print swarm[j].position_i
			i+=1

		# print final results
		print('\nFINAL SOLUTION:')
		print(pos_best_g)
		print(err_best_g)
		for i in range(0,num_particles):
			X_PSO.append(swarm[i].position_i)
		#return X_PSO

if __name__ == "__PSO__":
	main()

#--- RUN ----------------------------------------------------------------------+

initial=[[5,4,3],[1,2,0]]               # initial starting location [x1,x2...]
#bounds=[(-10,10),(-10,10)]  # input bounds [(x1_min,x1_max),(x2_min,x2_max)...]


for item in initial:
	for i in range(0,len(item)):
		if item[i] == 0:
			item[i] = -1
	
print (initial)

PSO(func1, initial, num_particles=len(initial), maxiter=10, verbose=True)
print (X_PSO)

#--- END ----------------------------------------------------------------------+
