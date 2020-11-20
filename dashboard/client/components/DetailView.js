import { useState } from 'react';
import { Progress, Result, Image, List, Card, Slider, Collapse, Radio, Tooltip, Table, Tag, Space } from 'antd';
import { ShrinkOutlined, AreaChartOutlined, SlidersOutlined } from '@ant-design/icons';
import { observable, action, makeObservable, computed } from 'mobx';
import { observer } from "mobx-react"

import  colormap from 'colormap';

import { range, every, uniq, xor, min, max } from 'lodash';

import dm from '../models/DataManager';

const { Column, ColumnGroup } = Table;
const { Panel } = Collapse;

const CORRECT_COLOR = '#32a852';
const INCORRECT_COLOR = '#a83232';


const colors = colormap({
    colormap: [{index: 0, rgb: [168, 50, 50]}, {index: 1, rgb: [50, 168, 82]}],
    nshades: 101,
    format: 'hex',
    alpha: 1
});


const nn = (array, value) => {
  const distances = array.map(x => Math.abs(value - x));
  const best_distance = min(distances);
  return distances.indexOf(best_distance);
};

class State {

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
    let counter = 0;
    for (const d of data) {
      if (d[d.length - 1]) {
        counter++;
      }
    }
    return counter / data.length;
  }
}

const ModeSelector = observer(({ record, currentState }) => {
  let current_mode = currentState.modes[record.key];
  const isControl = typeof(record.children) != 'undefined';

  if (isControl) {
    const childrenModes = record.children.map(({ key }) => currentState.modes[key]);
    if ((new Set(childrenModes)).size == 1) {
      current_mode = childrenModes[0];
    }
  }

  const allowHeatmap = (!isControl && Object.values(currentState.modes).filter(x=> x === 'heatmap').length < 2) || current_mode === 'heatmap'

  return <Radio.Group
    optionType="button"
    buttonStyle="solid"
    size='medium'
    onChange={action(e => {
      const v = e.target.value;
      if (isControl) {
        for (const k of Object.keys(currentState.modes)) {
          if (k.startsWith(record.key)) {
            currentState.modes[k] = v
          }
        }
      } else {
        currentState.modes[record.key] = v;
      }
    })}
    value={current_mode}
  >
      <Radio.Button value="heatmap" size='large' disabled={!allowHeatmap}>
        <Tooltip title="Heat map parameter">
          <AreaChartOutlined />
        </Tooltip>
      </Radio.Button>
      <Radio.Button value="slider">
        <Tooltip title="Select value with slider">
          <SlidersOutlined />
        </Tooltip>
      </Radio.Button>
      <Radio.Button value="aggregate">
        <Tooltip title="Aggregate all samples">
          <ShrinkOutlined />
        </Tooltip>
      </Radio.Button>
  </Radio.Group>
})

const SliderControl = observer(({ record, currentState }) => {
  const isControl = typeof(record.children) != 'undefined';
  if (isControl) {
    return null;
  }

  const { key } = record;

  if (currentState.modes[key] !== 'slider') {
    return null;
  }
  const ix = dm.data.parameters.indexOf(key);
  const all_values = dm.possibleValues[ix]
  const max_value = max(all_values);
  const min_value = min(all_values);
  const marks = Object.fromEntries(all_values.map(x => [x, '']))


  return <Slider value={currentState.sliderValues[key]}
    marks={marks} step={null} min={min_value} max={max_value}
    onChange={action((x) => {
      currentState.sliderValues[key] = x;
  })}/>
});

const RenderControls = ({ currentState }) => {

  const dd = dm.data;
  const parameters = dm.data.parameters;
  const pre_result = {}

  const all_controls = Object.values(parameters).filter(key => {
    const ix = dm.data.parameters.indexOf(key);
    const all_values = dm.possibleValues[ix]
    return all_values.length > 1;
  });


  for (const parameter of all_controls) {
    const [control_name, param_name] = parameter.split('.')
    if (typeof(pre_result[control_name]) === 'undefined') {
      pre_result[control_name] = {
        key: control_name,
        control: control_name.replace('Control', ''),
        param: '',
        children: []
      };
    }

    pre_result[control_name].children.push({
      key: parameter,
      control: '',
      param: param_name,
    })
  }

  const allParams = Object.values(pre_result)

  return <>
    <Table dataSource={allParams} bordered size="small" pagination={false}>
    <Column title="Control" dataIndex="control" key="control" width="175px"/>
    <Column title="Parameter" dataIndex="param" key="param" width="100px"/>
    <Column title="Mode" dataIndex="age" key="age" width="160px"
      render={(_, record) => <ModeSelector record={record} currentState={currentState} />}
    />
    <Column title="Filter" key="address"
      render={(_, item) => <SliderControl record={item} currentState={currentState} />}
    />
  </Table>
  </>
}



const RenderImages = observer(({ currentState }) => {
  const selectedSamples = currentState.selectedSamplesWHeatmap
  return <>
    <List
      grid={{ gutter: 15, column: 6 }}
      dataSource={selectedSamples}
      pagination={{
        pageSize: 6*6,
        position: 'top'
      }}
      renderItem={item => {
        const is_correct = item[item.length - 1]
        let style = {
          width: '100%',
          border: '10px solid'
        };

        if (is_correct) {
          style['borderColor'] = CORRECT_COLOR
        } else {
          style['borderColor'] = INCORRECT_COLOR
        }

        return <>
          <List.Item style={{ marginTop: '15px', marginBottom: '0' }}>
            <Image style={style} src={new URL(`images/${item[item.length - 4]}`, dm.currentUrl).toString()} />
          </List.Item>
        </>;
      }}
    />
  </>

  return selectedSamples.length;
});

const HeatMap = observer(({ currentState }) => {
  const { heatMapAxes } = currentState;

  if (heatMapAxes.length != 2) {
    return <Result
        status="error"
        title="Error"
        subTitle="You need to have exactly two parameters in heatmap mode to enable this display"
    />
  }

  const data = currentState.heatMapData;

  const [ x_bins, y_bins] = currentState.heatMapBins;

  const binCounts = x_bins.map(x => y_bins.map(y => 0));
  const correctCounts = x_bins.map(x => y_bins.map(y => 0));

  for (const [x_value, y_value, is_correct] of data) {
    const x_bin = nn(x_bins, x_value);
    const y_bin = nn(y_bins, y_value);
    binCounts[x_bin][y_bin]++;
    if (is_correct) {
      correctCounts[x_bin][y_bin]++;
    }
  }

  return <>
    <div style={{ width:'100%', height:'100%',  backgroundColor: 'white',
                  padding: '15px 15px 20px 30px'
    }}>
      <table style={{ width: '100%', height: '100%', borderSpacing: '3px', borderCollapse: 'separate',
        textAlign: 'center', color: 'white', fontWeight:'bold'
      }}>
        {y_bins.map((y_value, y) => (
          <tr key={`row${y}`}>
            {x_bins.map((x_value, x) => {
              const accuracy = correctCounts[x][y] / binCounts[x][y];
              return <>
                <Tooltip
                  arrowPointAtCenter
                  title={() => <>
                      {correctCounts[x][y]}/{binCounts[x][y]} ({(accuracy * 100).toFixed(1)}%) correct <br/>
                      {heatMapAxes[0].split('.')[1]}={x_value.toPrecision(6)} <br/>
                      {heatMapAxes[1].split('.')[1]}={y_value.toPrecision(6)} <br/>

                      </>}>
                  <td
                    style={{backgroundColor: colors[Math.round(accuracy * (colors.length - 1))]}}
                    key={`col${x}`}
                    width={`${100/x_bins.length}%`}
                    onMouseLeave={action(() => currentState.selectedHeatMapCell = null)}
                    onMouseEnter={action(() => {
                      currentState.selectedHeatMapCell = [x, y]
                    })}
                  >
                    {Math.round(accuracy * 100)}%
                  </td>
                </Tooltip>
              </>
            })}
          </tr>
        ))}
      </table>
      <div style={{ position: 'absolute', bottom: 0, width:'100%', textAlign:'center', height: '20px'}}>
        {heatMapAxes[0]}
      </div>
      <div style={{ position: 'absolute', top: 0, height:'100%', textAlign:'center', width: '20px', }}>
        <div style={{ position: 'absolute', bottom: '50%', transform: 'rotate(-90deg) translateY(-125px)',
          transformOrigin: '50% 0', width:'200px'
        }}>
        {heatMapAxes[1]}
        </div>
      </div>
    </div>
  </>
})

const CorrectBar = observer(({ currentState }) => {
  return (
    <div style={{ backgroundColor: 'white', padding: '15px 15px 15px 15px' }}>
      <Progress
        strokeWidth={20}
        percent={(currentState.filteredAccuracy * 100).toPrecision(2)}
        status="active"
      />
    </div>
  );
});

const Main = () => {
  const [ currentState, setCurrentState ] = useState(null);

  if (currentState === null) {
    setCurrentState(new State());
  } else {
    currentState.update();
  }
  return <>
    <div style={{ position: 'relative', backgroundColor: 'white' }}>
      <div style={{ width: '50%', display: 'inline-block', padding: '20px'}}>
        <RenderControls currentState={currentState} />
      </div>

      <div style={{ width: '50%', display: 'block', position:'absolute', paddingLeft: '10px', height: '100%', top: 0, right: 0}}>
        <HeatMap currentState={currentState} />
      </div>
    </div>
    <CorrectBar currentState={currentState} />
    <RenderImages currentState={currentState} />

  </>
}

export default Main;
