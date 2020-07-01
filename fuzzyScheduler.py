# #Joel Braganza z5283268

# The code in this file is written for the COMP9414 Assignment 1
# It schedules a set of tasks based on certain hard and soft constraints
# For this assignment AI Python was extended and overridden to write a fuzzy scheduler that solves an
#  arc-consistent search problem
# The problem operates over three constraints viz, unary constraints(hard), binary constraints and
#  soft constraints (not mandatory but incurr a cost)
# This is a constraint satisfaction problem and we use the CSP class of AI Python to execute it. 
# We also use AStarSearcher class that extends Searcher, for the priority queue to store the frontier
# 
# We also add a heuristic function to calculation to estimate the cost and then rewrite the Search_with_AC_from_CSP 
# as Search_with_AC_from_Cost_CSP to accout for the cost 


import sys
from cspConsistency import Search_with_AC_from_CSP
from searchGeneric import AStarSearcher

# Merely copying the AI Python Code into this file for easier readability 
class Constraint(object):
    """A Constraint consists of
    * scope: a tuple of variables
    * condition: a function that can applied to a tuple of values
    for the variables
    """
    def __init__(self, scope, condition):
        self.scope = scope
        self.condition = condition

    def __repr__(self):
        return self.condition.__name__ + str(self.scope)

    def holds(self,assignment):
        """returns the value of Constraint con evaluated in assignment.

        precondition: all variables are assigned in assignment
        """
        return self.condition(*tuple(assignment[v] for v in self.scope))

# Create CSP_cost from cspProblem.py that add satisfy soft_Consistency and deadline_costs
class CSP_cost(object):
   
    def __init__(self, domains, constraints, constraints_s, deadline_costs):

        """Here we consider domains as the 'variables' of or csp. Stored in a dictionary
            Constraints are stored as a list
        """
        self.variables = set(domains)
        self.domains = domains
        self.constraints = constraints
        self.var_to_const = {var:set() for var in self.variables}
        self.constraints_s= constraints_s
        self.deadline_costs = deadline_costs

        for con in constraints:
            for var in con.scope:
                self.var_to_const[var].add(con)

    def __str__(self):
        """string representation of CSP"""
        return str(self.domains)

    def __repr__(self):
        """more detailed string representation of CSP"""
        return "CSP("+str(self.domains)+", "+str([str(c) for c in self.constraints])+")"

    def consistent(self,assignment):
        """assignment is a variable:value dictionary
        returns True if all of the constraints that can be evaluated
                        evaluate to True given assignment.
        """
        return all(con.holds(assignment)
                    for con in self.constraints
                    if all(v in  assignment  for v in con.scope))

"""
Storing the domain as a dictionary with keys as the members of the domain and 
values and numbers for better serializability
Values in the domain dictionary are only concatenation of the days and time values 
"""
days = {'mon': '1', 'tue': '2', 'wed': '3', 'thu': '4', 'fri': '5'}
times = {'9am': '1', '10am': '2', '11am': '3', '12pm': '4', '1pm': '5', '2pm': '6', '3pm': '7', '4pm': '8', '5pm': '9'}
domain = {'11', '12', '13', '14', '15', '16', '17', '18', '19','21', '22', '23', '24', '25', '26', '27', '28', '29','31', '32', '33', '34', '35', '36', '37', '38', '39','41', '42', '43', '44', '45', '46', '47', '48', '49','51', '52', '53', '54', '55', '56', '57', '58', '59'}
time_duration = {}
task_vals = {}
constraints_h = []
constraints_s= {}
deadline_costs = {}

# implementing unary constraints on input data
def constraints_h_day(day):
    """is a value"""
    def is_val(x): return x[0]//10 == int(day)
    is_val.__name__ = day+"=="
    return is_val
    
def constraints_h_time(time):
    def is_val(x): return x[0]%10 == int(time)
    is_val.__name__ = time+"=="
    return is_val

def constraints_h_endsbefore_daytime(day,time):
    def is_val(x): return x[1] <= int(day + time)
    return is_val

def constraints_h_startsbefore_daytime(day,time):
    def is_val(x): return x[0] <= int(day + time)
    return is_val

def constraints_h_endsafter_daytime(day,time):
    def is_val(x): return x[1] >= int(day + time)
    return is_val


def constraints_h_startsafter_daytime(day,time):

    def is_val(x): return x[0] >= int(day + time)
    return is_val



def constraints_h_startin(day1,time1,day2,time2):
    def is_val(x):
        giventime1 = int(day1 + time1)
        giventime2 = int(day2 + time2)
        return x[0]>=giventime1 and x[0]<=giventime2
    return is_val

def constraints_h_endsbefore_time(time):
    def is_val(x): return x[1]%10 <= int(time)
    is_val.__name__ = time + "<="
    return is_val

def constraints_h_endin(day1,time1,day2,time2):
    def is_val(x): return x[1]>=int(day1 + time1) and x[1]<=int(day2 + time2)
    return is_val

def constraints_h_startsbefore_time(time):
    def is_val(x): return x[0]%10 <= int(time)
    is_val.__name__ = time + "<="
    return is_val


def constraints_h_startsafter_time(time):
    def is_val(x): return x[0]%10 >= int(time)
    is_val.__name__ = time + ">="
    return is_val

def constraints_h_endsafter_time(time):
    def is_val(x): return x[1]%10 >= int(time)
    is_val.__name__ = time + ">="
    return is_val


# identify tasks and their duration and set default domain
#  (super set of all domain entries)
def read_tasks_vals(line):
    if line[0] == 'task':
        time_duration[line[1]] = line[2]
        temp = set()
        duration = int(line[2])
        for str in domain:
            if int(str[1]) + duration <= 9:
                temp.add(int(str))
        task_vals[line[1]] = set((t, t + duration) for t in temp)
    else:
        pass
    return task_vals


# executing binary constraints 
def binary_constraints_before(cond_1,cond_2):
    return cond_1[1] <= cond_2[0]

def binary_constraints_after(cond_1,cond_2):
    return cond_2[1] <= cond_1[0]

def binary_constraints_sameday(cond_1,cond_2):
    return cond_1[0]//10 == cond_2[0]//10

def binary_constraints_startsat(cond_1,cond_2):
    return cond_1[0] == cond_2[1]


#reading the input file 
filename = sys.argv[1]
with open(filename,'r') as file:
    for line in file:
        line = line.strip()
        line = line.replace(',', '')
        line = line.replace('-', '')
        line = line.split(' ')
        if '#' in line:
            continue
        if line[0] == '':
            continue
        # reading input to get task and duration
        task_vals = read_tasks_vals(line)
        
        # reading input for constraints
        if line[0] == 'constraint':
            cond_1 = line[1]
            cond_2 = line[-1]
            if 'before' in line:
                constraints_h.append(Constraint((cond_1, cond_2), binary_constraints_before))
            if 'after' in line:
                constraints_h.append(Constraint((cond_1, cond_2), binary_constraints_after))
            if 'sameday' in line:
                constraints_h.append(Constraint((cond_1, cond_2), binary_constraints_sameday))
            if 'startsat' in line:
                constraints_h.append(Constraint((cond_1, cond_2), binary_constraints_startsat))
        
        # deadline extensions put in dict
        elif (line[0] == 'domain') and (line[2] =='endsby'):
            task = line[1]
            day = days[line[3]]
            time = times[line[4]]
            deadline_costs[task] = int(line[-1])
            constraints_s[task] = int(day + time)

        # reading hard constraints into a tuple
        elif len(line)==3:
            task = line[1]
            if (line[0] == 'domain') and (line[2] in days):
                day = days[line[2]]
                constraints_h.append(Constraint((task,), constraints_h_day(day)))
            elif (line[0] == 'domain') and (line[2] in times):
                time = times[line[2]]
                constraints_h.append(Constraint((task,), constraints_h_time(time)))
        elif len(line)>3:
            task = line[1]
            if (line[0] == 'domain') and (line[2] == 'startsbefore') and (line[3] in days):
                day = days[line[-2]]
                time = times[line[-1]]
                constraints_h.append(Constraint((task,), constraints_h_startsbefore_daytime(day, time)))
            elif (line[0] == 'domain') and (line[2] == 'startsafter') and (line[3] in days):
                time = times[line[-1]]
                day = days[line[-2]]
                constraints_h.append(Constraint((task,), constraints_h_startsafter_daytime(day, time)))
            elif (line[0] == 'domain') and (line[2] == 'endsbefore') and (line[3] in days):
                time = times[line[-1]]
                day = days[line[-2]]
                constraints_h.append(Constraint((task,), constraints_h_endsbefore_daytime(day, time)))
            elif (line[0] == 'domain') and (line[2] == 'endsafter') and (line[3] in days):
                day = days[line[-2]]
                time = times[line[-1]]
                constraints_h.append(Constraint((task,), constraints_h_endsafter_daytime(day, time)))
            elif (line[0] == 'domain') and (line[2] == 'startsin'):
                day1 = days[line[3]]
                #since some times are 2 digit and some are 1 digit
                if len(line[4])==6:
                    time=line[4][0:3]
                    day=line[4][3:6]
                else:
                    time=line[4][0:4]
                    day=line[4][4:7]
                time1 = times[time]
                day2 = days[day]
                time2 = times[line[5]]
                constraints_h.append(Constraint((task,), constraints_h_startin(day1, time1, day2, time2)))
            elif (line[0] == 'domain') and (line[2] == 'endsin'):
                day1 = days[line[3]]
                #since some times are 2 digit and some are 1 digit
                if len(line[4])==6:
                    time=line[4][0:3]
                    day=line[4][3:6]
                else:
                    time=line[4][0:4]
                    day=line[4][4:7]
                time1 = times[time]
                day2 = days[day]
                time2 = times[line[5]]
                constraints_h.append(Constraint((task,), constraints_h_endin(day1, time1, day2, time2)))
            elif (line[0] == 'domain') and (line[2] == 'startsbefore') and (line[3] in times):
                if len(line) == 5:
                    time = times[line[-1]]
                    constraints_h.append(Constraint((task,), constraints_h_startsbefore_time(time)))
            elif (line[0] == 'domain') and (line[2] == 'endsbefore') and (line[3] in times):
                time = times[line[-1]]
                constraints_h.append(Constraint((task,), constraints_h_endsbefore_time(time)))
            elif (line[0] == 'domain') and (line[2] == 'startsafter') and (line[3] in times):
                time = times[line[-1]]
                constraints_h.append(Constraint((task,), constraints_h_startsafter_time(time)))
            elif (line[0] == 'domain') and (line[2] == 'endsafter') and (line[3] in times):
                time = times[line[-1]]
                constraints_h.append(Constraint((task,), constraints_h_endsafter_time(time)))

# rewriting the function to include cost and deadlines extensions
class Search_with_AC_from_Cost_CSP(Search_with_AC_from_CSP):
    def __init__(self,csp):
        super().__init__(csp)
        self.cost = []
        self.constraints_s= csp.constraints_s
        self.deadline_costs = deadline_costs

    # this is the heuristic function that estimates the minimum cost 
    def heuristic(self, node):
        minCost = []
        for task_num in node:
            if task_num in self.constraints_s:
                list_Cost = []
                last_time = int(self.constraints_s[task_num])
                for n in node[task_num]:
                    if n[1] > last_time:
                        if (n[1]//10- last_time//10)==0:
                            min_cost = ((n[1]%10) - (last_time%10))
                        else:
                            min_cost_one_day = ((n[1] % 10) - (last_time % 10))
                            min_cost_other_day = (n[1]//10- last_time//10) *24
                            min_cost = min_cost_one_day + min_cost_other_day
                        list_Cost.append(self.deadline_costs[task_num] * min_cost)
                    else: list_Cost.append(0)
                if len(list_Cost) != 0:
                    minCost.append(min(list_Cost))
        return sum(minCost)

# displaying to stdout as required by assignment spec
def print_output(Solver,searchProblem):
    if Solver is not None:
        s_best=Solver.end()
        for task in Solver.end():
            day = str(list(s_best[task])[0][0])[0]
            time = str(list(s_best[task])[0][0])[1]
            for day_key in days:
                if days[day_key] == day:
                    day = day_key
            for time_key in times:
                if times[time_key] == time:
                    time = time_key
            print(f'{task}:{day} {time}')
        print(f'cost:{searchProblem.heuristic(s_best)}')
    else:
        print('No solution')

CSP_Cost_1 = CSP_cost(task_vals,constraints_h,constraints_s,deadline_costs)
searchProblem = Search_with_AC_from_Cost_CSP(CSP_Cost_1)
Solver = AStarSearcher(searchProblem).search()
print_output(Solver,searchProblem)