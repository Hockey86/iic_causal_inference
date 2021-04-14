#!/bin/bash
PYTHON=python3
FLAG=-w ignore
FILENAME=fit_simulator.py
$PYTHON $FLAG $FILENAME baseline
$PYTHON $FLAG $FILENAME normal_expit_ARMA 1 6
$PYTHON $FLAG $FILENAME cauchy_expit_ARMA 1 6
$PYTHON $FLAG $FILENAME cauchy_expit_lognormal_ARMA 1 6
$PYTHON $FLAG $FILENAME cauchy_expit_lognormal_drugoutside_ARMA 1 6
$PYTHON $FLAG $FILENAME cauchy_expit_a0_as_lognormal_drugoutside_ARMA 1 6
$PYTHON $FLAG $FILENAME cauchy_expit_lognormal_drugoutside_ARMA 2 6
$PYTHON $FLAG $FILENAME cauchy_expit_lognormal_drugoutside_ARMA 3 6
