import { computed, action, makeObservable, observable } from 'mobx';
import { range, sortedUniq } from 'lodash';

class DataManager {
  loaded = false;
  failed = false;
  currentUrl = null;
  data = {};

  constructor() {
    makeObservable(this, {
      loaded: observable,
      failed: observable,
      currentUrl: observable,
      data: observable.ref,
      possibleValues: computed,
      fetchUrl: action,
      setLoaded: action
    });
  }

  async fetchUrl(newUrl) {
    if(newUrl === this.currentUrl) {
      return; // Nothing to do here, we already have the data (or it's already loading)
    }

    this.loaded = false;
    this.currentUrl = newUrl;

    try {
      await new Promise(r => setTimeout(r, 2000));
      const request = await fetch(newUrl);
      this.failed = false;
      this.setLoaded(await request.json());
    } catch(e) {
      console.log(e);
      this.failed = true;
    }
  }

  setLoaded(data) {
      this.data = data;
      this.loaded = true;
  }

  get possibleValues() {
    const n = this.data.data[0].length;
    const result = [];

    for (const i of range(n)) {
      const all_values = sortedUniq(this.data.data.map(x => x[i]));
      result.push(all_values);
    }

    return result
  }
}

export default new DataManager;
