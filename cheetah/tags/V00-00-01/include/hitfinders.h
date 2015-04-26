#include "cheetah/cheetahmodules.h"

#ifndef HITFINDERS_H
#define HITFINDERS_H

void integratePixAboveThreshold(float *data,uint16_t *mask,long pix_nn,float ADC_threshold,uint16_t pixel_options,long *nat,float *tat);
int hitfinder1(cGlobal *global, cEventData *eventData, long detID);
int hitfinder2(cGlobal *global, cEventData *eventData, long detID);
int hitfinder4(cGlobal *global, cEventData *eventData, long detID);
int hitfinder9(cGlobal *global, cEventData *eventData);
int hitfinderTOF(cGlobal *global, cEventData *eventData);
int hitfinderProtonsandPhotons(cGlobal *global, cEventData *eventData, long detID);
bool containsEvent(std::string event, cGlobal *global);

#endif
