# Uncertainty Design

Runtime uncertainty is a core feature, not setup noise.

- Weather is sampled every quarter from Cambridge climate-normal-scaled seasonal distributions, anchored to historical quarterly records from `2000-2023`.
- Prices are resampled every quarter around DEFRA-derived crop and input anchors.
- Pest pressure is stochastic every quarter and depends on the realised weather regime plus rotation pressure.
- Soil evolves with mild subcomponent noise each step, so even similar plans can separate over long horizons.

Task generation also varies scenario parameters, but the main benchmark path remains uncertain during the live episode.
