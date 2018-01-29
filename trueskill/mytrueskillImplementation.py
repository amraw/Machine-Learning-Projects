####################
# Name: Amit Rawat #
# UIN:- 326005672  #
###################

import numpy as np
import math as mt
from scipy.stats import norm

#Class for Gaussain object
class Gaussian(object):
    pi = 0
    tau = 0

    def __init__(self, mu=None, sigma=None, tau=None, pi=None):
        if pi is not None:
            self.pi = pi
            self.tau = tau
        elif mu is not None:
            self.pi = sigma ** -2
            self.tau = self.pi * mu
        else:
            self.pi = 0
            self.tau = 0


    def getmusigma(self):
        if self.pi == 0.0:
            return 0, float("inf")
        else:
            return self.tau / self.pi, mt.sqrt(1 / self.pi)

    def __mul__(self, other):
        newpi = self.pi + other.pi
        newtau = self.tau + other.tau
        return Gaussian(pi=newpi, tau=newtau)

    def __div__(self, other):
        newpi = self.pi - other.pi
        newtau = self.tau - other.tau
        return Gaussian(pi=newpi, tau=newtau)
    __truediv__ = __div__

    def __str__(self):
        if self.pi == 0.0:
            return "N(mu=0.0,sigma=inf)"
        else:
            template = 'N(mu={0:.3f},sigma={1:.3f})'
            return template.format(self.tau / self.pi, mt.sqrt(1 / self.pi))

    def __repr__(self):
        return "N(pi={0.pi},tau={0.tau})".format(self)

#calculated draw probability from draw margin
def calculateDrawProb(drawMargin, beta, players_count=2):
    return 2 * norm.cdf(drawMargin / (mt.sqrt(players_count) * beta)) - 1

#Calculate Draw margin from draw probability
def calculateDrawMargin(drawProb, beta, players_count=2):
    return norm.ppf((drawProb + 1.0) / 2) * mt.sqrt(players_count) * beta

# Default initial mean and sigma
DEFAULT_INITIAL_MEAN = 25.0
DEFAULT_INTIAL_SIGMA = DEFAULT_INITIAL_MEAN / 3.0

# Function for assigning the Game related parameters
def GameInformation(beta=None, drawMargin = None,drawProbability=None, dynamicFactor=None):
    global BETA, DRAWMARGIN, DYNAMICFACTOR
    if dynamicFactor is None:
        DYNAMICFACTOR = DEFAULT_INITIAL_MEAN / 300.0
    else:
        DYNAMICFACTOR = dynamicFactor

    if beta is None:
        BETA = DEFAULT_INTIAL_SIGMA / 2.0
    else:
        BETA = beta
    if drawMargin is None:
        if drawProbability is None:
            drawProbability = 0.10
        DRAWMARGIN = calculateDrawMargin(drawProbability, BETA)
    else:
        DRAWMARGIN = drawMargin

# Calculation of multiplicative correction tern
def vFwin(teamPerformanceDifference, drawMargin):
    return norm.pdf(teamPerformanceDifference - drawMargin) / norm.cdf(teamPerformanceDifference - drawMargin)


def wFwin(teamPerformanceDifference, drawMargin):
    return vFwin(teamPerformanceDifference, drawMargin) * (
    vFwin(teamPerformanceDifference, drawMargin) + teamPerformanceDifference - drawMargin)


def vFdraw(teamPerformanceDifference, drawMargin):
    return (norm.pdf(-drawMargin - teamPerformanceDifference) - norm.pdf(drawMargin - teamPerformanceDifference)) / (
    norm.cdf(drawMargin - teamPerformanceDifference) - norm.cdf(-drawMargin - teamPerformanceDifference))


def wFdraw(teamPerformanceDifference, drawMargin):
    return vFdraw(teamPerformanceDifference, drawMargin) **2 + \
           (((drawMargin - teamPerformanceDifference) * norm.pdf(drawMargin - teamPerformanceDifference) +
             (drawMargin + teamPerformanceDifference) * norm.pdf(drawMargin + teamPerformanceDifference)) / (
            norm.cdf(drawMargin - teamPerformanceDifference) -
            norm.cdf(-drawMargin - teamPerformanceDifference)))

# Class for creating variable node
class VariableNode(object):

  def __init__(self):
    self.nodeValue = Gaussian()
    self.factorsConnected = {}

  def ConnectToFactor(self, factorNode):
        self.factorsConnected[factorNode] = Gaussian()

  def GetMessage(self, factorNode):
        return self.factorsConnected[factorNode]

  def UpdateMessage(self, factorNode, message):
        prevMessage = self.factorsConnected[factorNode]
        self.nodeValue = self.nodeValue / prevMessage * message
        self.factorsConnected[factorNode] = message

  def UpdateValue(self, factorNode, value):
        prevMessage = self.factorsConnected[factorNode]
        self.factorsConnected[factorNode] = value * prevMessage / self.nodeValue
        self.nodeValue = value

# Base class for factor node
class FactorNode(object):
    def __init__(self, variables):
        self.variables = variables
        for varNode in variables:
            varNode.ConnectToFactor(self)

# Factor node for calculating the prior
class PriorFactorNode(FactorNode):
    def __init__(self, variable, param):
        super(PriorFactorNode, self).__init__([variable])
        self.param = param

    def begin(self):
        self.variables[0].UpdateValue(self, self.param)

# Factor node for calculating the likelihood factor
class LikelihoodFactor(FactorNode):
    def __init__(self, meanVariable, valueVariable, variance):
        super(LikelihoodFactor, self).__init__([meanVariable, valueVariable])
        # FactorNode.__init__([meanVariable, valueVariable])
        self.mean = meanVariable
        self.value = valueVariable
        self.variance = variance

    def UpdateValue(self):
        y = self.mean.nodeValue
        fy = self.mean.GetMessage(self)
        a = 1.0 / (1.0 + self.variance * (y.pi - fy.pi))
        self.value.UpdateMessage(self, Gaussian(pi=a*(y.pi - fy.pi),
                                                tau=a*(y.tau - fy.tau)))

    def UpdateMean(self):
        x = self.value.nodeValue
        fx = self.value.GetMessage(self)
        a = 1.0 / (1.0 + self.variance * (x.pi - fx.pi))
        self.mean.UpdateMessage(self, Gaussian(pi=a*(x.pi - fx.pi),
                                               tau=a * (x.tau - fx.tau)))

# Factor node
class FactorNodeSumCal(FactorNode):
    def __init__(self, sumVar, termVar, coeffs):
        super(FactorNodeSumCal, self).__init__([sumVar] + termVar)
        # FactorNode.__init__([sumVar]+termVar)
        self.sum = sumVar
        self.terms = termVar
        self.coeffs = coeffs

    def UpdateNode(self, var, y, fy, a):
        new_pi = 1.0 / (sum(a[j] ** 2 / (y[j].pi - fy[j].pi) for j in range(len(a))))
        new_tau = new_pi * sum(a[j] *
                               (y[j].tau - fy[j].tau) / (y[j].pi - fy[j].pi)
                               for j in range(len(a)))
        var.UpdateMessage(self, Gaussian(pi=new_pi, tau=new_tau))

    def UpdateSum(self):
        y = [t.nodeValue for t in self.terms]
        fy = [t.GetMessage(self) for t in self.terms]
        a = self.coeffs
        self.UpdateNode(self.sum, y, fy, a)

    def UpdateTerms(self, index):
        b = self.coeffs
        a = [-b[k] / b[index] for k in range(len(b)) if k != index]
        a.insert(index, 1.0 / b[index])
        v = self.terms[:]
        v[index] = self.sum
        y = [i.nodeValue for i in v]
        fy = [i.GetMessage(self) for i in v]
        self.UpdateNode(self.terms[index], y, fy, a)

# Truncating factor ndoe
class FactorEnd(FactorNode):
    def __init__(self, variable, V, W, drawMargin):
        super(FactorEnd, self).__init__([variable])
        # FactorNode.__init__([variable])
        self.var = variable
        self.V = V
        self.W = W
        self.drawMargin = drawMargin

    def Update(self):
        x = self.var.nodeValue
        fx = self.var.GetMessage(self)
        c = x.pi - fx.pi
        d = x.tau - fx.tau
        Vf = self.V(d / mt.sqrt(c), self.drawMargin * mt.sqrt(c))
        Wf = self.W(d / mt.sqrt(c), self.drawMargin * mt.sqrt(c))
        newVal = Gaussian(pi=(c / (1.0 - Wf)), tau=(d + mt.sqrt(c) * Vf) / (1 - Wf))
        self.var.UpdateValue(self, newVal)

# Function which do message passing
def functionToUpdateValue(playerInfo):
    playersInfonew = sorted(playerInfo, key=lambda x: x.rank)

    skillNodes = [VariableNode() for player in playersInfonew]
    performanceNodes = [VariableNode() for player in playersInfonew]
    teamPerformanceNodes = [VariableNode() for player in playersInfonew]
    differenceNodes = [VariableNode() for player in playersInfonew]
    playerPriorSkill = [PriorFactorNode(skill, Gaussian(mu=player.skill[0],
                                                        sigma=mt.sqrt(player.skill[1] ** 2 + DYNAMICFACTOR ** 2)))
                        for (skill, player) in zip(skillNodes, playersInfonew)]
    skillToPerformance = [LikelihoodFactor(skill, performance, BETA ** 2) for (skill, performance) in
                          zip(skillNodes, performanceNodes)]

    performanceToTeam = [FactorNodeSumCal(team, [player], [1]) for (player, team) in
                         zip(performanceNodes, teamPerformanceNodes)]

    teamDifference = [FactorNodeSumCal(diff, [team1, team2], [+1,- 1]) for (diff, team1, team2) in
                      zip(differenceNodes, teamPerformanceNodes[:-1], teamPerformanceNodes[1:])]

    endNode = [FactorEnd(diff, vFdraw if player1.rank == player2.rank else vFwin,
                         wFdraw if player1.rank == player2.rank else wFwin, DRAWMARGIN)
               for (diff, player1, player2) in zip(differenceNodes, playersInfonew[:-1], playersInfonew[1:])]

    for playerSkill in playerPriorSkill:
        playerSkill.begin()
    for sktoPerform in skillToPerformance:
        sktoPerform.UpdateValue()
    for pertoteam in performanceToTeam:
        pertoteam.UpdateSum()

    for j in range(7):
        for diff in teamDifference:
            diff.UpdateSum()
        for end in endNode:
            end.Update()
        for diff in teamDifference:
            diff.UpdateTerms(0)
            diff.UpdateTerms(1)

    for teamToPerform in performanceToTeam:
        teamToPerform.UpdateTerms(0)
    for performToSkill in skillToPerformance:
        performToSkill.UpdateMean()
    for skill, player in zip(skillNodes, playersInfonew):
        player.skill = skill.nodeValue.getmusigma()


class Player(object):
    pass


Ram = Player()
Ram.skill = (25.0, 25.0 / 3.0)

Shyam = Player()
Shyam.skill = (25.0, 25.0 / 3.0)

Geeta = Player()
Geeta.skill = (25.0, 25.0 / 3.0)

Sita = Player()
Sita.skill = (25.0, 25.0 / 3.0)

Ram.rank = 1
Shyam.rank = 2
Geeta.rank = 2
Sita.rank = 4
# Function for calculating the skill
GameInformation()
functionToUpdateValue([Ram, Shyam, Geeta, Sita])
# New calculated skill printed
print(" Ram: mu={0[0]:.3f}  sigma={0[1]:.3f}".format(Ram.skill))
print("   Shyam: mu={0[0]:.3f}  sigma={0[1]:.3f}".format(Shyam.skill))
print(" Geeta: mu={0[0]:.3f}  sigma={0[1]:.3f}".format(Geeta.skill))
print("Sita: mu={0[0]:.3f}  sigma={0[1]:.3f}".format(Sita.skill))