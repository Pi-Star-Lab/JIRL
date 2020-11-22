# A Joint Imitation-Reinforcement Learning Framework for Reduced Baseline Regret

Official Repository for "A Joint Imitation-Reinforcement Learning (JIRL) Framework for Reduced Baseline Regret"

## Technical Report

The report contains a detailed description of the experimental settings and hyperparameters used to obtain the results reported in our paper.

<object data="report/jirl-technical-report.pdf" type="application/pdf" width="750px" height="750px">
    <embed src="report/jirl-technical-report.pdf" type="application/pdf"></embed>
</object>

## Objectives
1. Leveraging a baseline’s online demonstrations to minimize the regret w.r.t the baseline policy during training
2. Eventually surpassing the baseline performance

## Assumptions
1. Access to a baseline policy at every time step
2. Uses an off-policy RL algorithm

## Framework

<p align="center">
  <img src="report/figures/jirl-flow.png" width="600">
</p>

## Experiment Domains

<div class="row">
  <div class="column">
    <img src="report/figures/pendulum-env.png" width="240">
    <div class="caption">Inverted pendulum</div>
  </div>
  <div class="column">
    <img src="report/figures/lander-env.png" width="240">
    <div class="caption">Lunar lander</div>
  </div>
  <div class="column">
    <img src="report/figures/walker-env.png" width="240">
    <div class="caption">Walker-2D</div>
  </div>
</div>

<div class="row">
  <div class="column">
    <img src="report/figures/carla-env.png" width="240">
    <div class="caption">Lane following (CARLA)</div>
  </div>
  <div class="column">
    <img src="report/figures/track_car.jpeg" width="240">
    <div class="caption">Lane following (JetRacer)</div>
  </div>
</div>

## Results
### Performance

<div class="row">
  <div class="column">
    <img src="report/figures/pendulum.png" width="240">
    <div class="caption">Inverted pendulum</div>
  </div>
  <div class="column">
    <img src="report/figures/lander-new-2.png" width="240">
    <div class="caption">Lunar lander</div>
  </div>
  <div class="column">
    <img src="report/figures/carla-new-2.png" width="240">
    <div class="caption">Lane following (CARLA)</div>
  </div>
</div>

<div class="row">
  <div class="column">
    <img src="report/figures/walker-new-2.png" width="240">
    <div class="caption">Walker-2D</div>
  </div>
  <div class="column">
    <img src="report/figures/jet-car.png" width="240">
    <div class="caption">Lane following (JetRacer)</div>
  </div>
</div>

### Baseline Regret

<div class="row">
  <div class="column">
    <img src="report/figures/baseline-regret.png" width="400">
  </div>
  <div class="column">
    <img src="report/figures/lander-trpo.png" width="300">
    <div class="caption">Lunar lander (JIRL vs TRPO)</div>
  </div>
</div>
