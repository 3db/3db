import { range, every, uniq, xor, min, max} from 'lodash';
import { observable, action, makeObservable, computed } from 'mobx';

import dm from '../models/DataManager';

export default class State {

  last_params = new Set()
  modes = {}
  sliderValues = {};
  selectedHeatMapCell = null;

  constructor() {
    makeObservable(this, {
      update: action,
      resetState: action,
      modes: observable,
      sliderValues: observable,
      selectedHeatMapCell: observable,
      selectedSamples: computed,
      selectedSamplesWHeatmap: computed,
      heatMapAxes: computed,
      heatMapAxesIx: computed,
      heatMapData: computed,
      heatMapBins: computed
    })
  }

  get heatMapAxes() {
    return Object.entries(this.modes).filter(([param, mode]) => mode === 'heatmap').map(x => x[0])
  }

  get heatMapAxesIx() {
    return this.heatMapAxes.map(x => dm.data.parameters.indexOf(x));
  }

  get heatMapData() {
    const ixes = this.heatMapAxesIx;
    const values = this.selectedSamples

    return values.map(x => [x[ixes[0]], x[ixes[1]], x[x.length - 1]]);
  }

  resetState() {
    const parameters = dm.data.parameters;
    this.modes = Object.fromEntries(parameters.map(x => [x, 'aggregate']))
    this.sliderValues = Object.fromEntries(parameters.map(x => {
      const ix = dm.data.parameters.indexOf(x);
      return [x, dm.possibleValues[ix][0]];
    }));
  }

  update() {
    const parameters = dm.data.parameters;
    const control_set = new Set(Object.values(parameters).map(x => x.split('.')[0]))
    if (xor(Array.from(control_set), Array.from(this.last_params)).length !== 0) {
      this.last_params = control_set;
      this.resetState();
    }
  }

  get selectedSamples() {
    return dm.data.data.filter(sample => every(dm.data.parameters, (key, ix) => {
      if (dm.possibleValues[ix].length == 1) { return true; }
      if (this.modes[key] == 'aggregate') { return true }
      if (this.modes[key] == 'heatmap') { return true }
      if (this.modes[key] == 'slider' && sample[ix] == this.sliderValues[key]) {
        return true
      }
      return false;
    }));
  }

  get selectedSamplesWHeatmap() {
    const samples = this.selectedSamples;
    if (this.selectedHeatMapCell == null) {
      return samples;
    }
    else {
      const hmCoords = this.sampleHeatMapCoords;
      return samples.filter((_, i) => hmCoords[i][0] === this.selectedHeatMapCell[0] && hmCoords[i][1] === this.selectedHeatMapCell[1]);
    }
  }

  get heatMapBins() {
    const data = this.heatMapData;
    const compute_bins = (ax) => {
      const values = data.map(x => x[ax]);
      const unique_values = uniq(values)
      unique_values.sort();
      const max_num_bins = 10;
      if (unique_values.length > max_num_bins) {
        const [ low_val, high_val ] = [min(unique_values), max(unique_values)];
        const step = (high_val - low_val) / (max_num_bins - 1);
        return range(low_val, high_val + step, step)
      } else {
        return unique_values
      }
    };

    return [compute_bins(0), compute_bins(1)];
  }

  get sampleHeatMapCoords() {
    const [ x_bins, y_bins ] = this.heatMapBins;
    return this.heatMapData.map(([x_value, y_value, is_correct]) => [nn(x_bins, x_value), nn(y_bins, y_value)])
  }

  get filteredAccuracy() {
    const data = this.selectedSamplesWHeatmap;
    const ix = dm.data.parameters.indexOf('is_correct');
    let counter = 0;
    for (const d of data) {
      if (d[ix]) {
        counter++;
      }
    }
    return counter / data.length;
  }
}