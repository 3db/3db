import { useState } from 'react';
import { Image, List, Card, Slider, Collapse, Radio, Tooltip, Table, Tag, Space } from 'antd';
import { ShrinkOutlined, AreaChartOutlined, SlidersOutlined } from '@ant-design/icons';
import { observable, action, makeObservable } from 'mobx';
import { observer } from "mobx-react"

import { every, uniq, xor, min, max } from 'lodash';

import dm from '../models/DataManager';

const { Column, ColumnGroup } = Table;
const { Panel } = Collapse;

class State {

  last_params = new Set()
  modes = {}
  sliderValues = {};

  constructor() {
    makeObservable(this, {
      update: action,
      resetState: action,
      modes: observable,
      sliderValues: observable
    })
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
    <Column title="Control" dataIndex="control" key="control" width="200px"/>
    <Column title="Parameter" dataIndex="param" key="param" width="100px"/>
    <Column title="Mode" dataIndex="age" key="age" width="160px"
      render={(_, record) => <ModeSelector record={record} currentState={currentState} />}
    />
    <Column title="Slider" key="address"
      render={(_, item) => <SliderControl record={item} currentState={currentState} />}
    />
  </Table>
  </>
}

const RenderImages = observer(({ currentState }) => {
  const entries = dm.data.parameters;
  const selectedSamples = dm.data.data.filter(sample => every(dm.data.parameters, (key, ix) => {
    if (dm.possibleValues[ix].length == 1) { return true; }
    if (currentState.modes[key] == 'aggregate') { return true }
    if (currentState.modes[key] == 'slider' && sample[ix] == currentState.sliderValues[key]) {
      return true
    }
    return false;
  }));
  return <>
    <List
      grid={{ gutter: 0, column: 6 }}
      dataSource={selectedSamples}
      pagination={{
        pageSize: 6*6,
        position: 'top'
      }}
      renderItem={item => (
        <List.Item>
          <Image style={{width: '100%'}} src={new URL(`images/${item[item.length - 4]}`, dm.currentUrl).toString()} />
        </List.Item>
      )}
    />
  </>

  return selectedSamples.length;
});

export default () => {
  const [ currentState, setCurrentState ] = useState(null);

  if (currentState === null) {
    setCurrentState(new State());
  } else {
    currentState.update();
  }
  return <>
    <RenderControls currentState={currentState} />
    <RenderImages currentState={currentState} />

  </>
}
