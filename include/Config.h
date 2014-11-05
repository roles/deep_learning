#ifndef _CONFIG_H
#define _CONFIG_H

const int maxLayer = 5;
const int maxUnit = 2000;
const int maxBatchSize = 20;

const double L2Reg = 0.0001;

typedef double RBMBuffer[maxUnit*maxBatchSize];

#endif
