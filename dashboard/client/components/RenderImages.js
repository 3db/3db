import { List } from 'antd';
import { fromPairs, zip} from 'lodash';
import { observer } from "mobx-react"

import { CORRECT_COLOR, INCORRECT_COLOR} from '../style/colors';

import dm from '../models/DataManager';
import { argMax } from '../utils/utils';

function dataForItem(item) {
    return fromPairs(zip(dm.data.parameters, item));
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
        const data = dataForItem(item);
        const is_correct = data["is_correct"];
        let style = {
          width: '100%',
        };

        let itemStyle = {
            position: 'relative',
            boxSizing: 'content-box',
            borderWidth: '10px',
            borderStyle: 'solid',
            borderColor: 'black'
        };
        if (is_correct) {
          itemStyle['borderColor'] = CORRECT_COLOR
        } else {
          itemStyle['borderColor'] = INCORRECT_COLOR
        }

        const rectStyle = {fill: 'rgba(0,0,0,0)', strokeWidth: '3', stroke: 'rgb(0,0,0)'};
        var rects = [];
        var texts = [];
        
        if(data["output_type"] == "bboxes") {
            const bboxes = data["outputs"];
            for(let bbox of bboxes) {
                if(bbox[0] < 0) { break; }
                const rand = Math.random().toString().substr(2, 8);
                rects.push(<rect x={(bbox[0] * 100) + "%"} 
                                y={(bbox[1] * 100) + "%"} 
                                width={((bbox[2] - bbox[0]) * 100) + "%"} 
                                height={((bbox[3] - bbox[1]) * 100) + "%"} 
                                style={rectStyle}
                                key={rand + "_rect"}/>)
                texts.push(<text x={(bbox[0] * 100) + "%"}
                                y={(bbox[1] * 100) + "%"}
                                fontSize="8"
                                style={{stroke: 'white', strokeWidth: '0.6em'}}
                                transform="translate(0 10)"
                                key={rand + "_text"}>{COCOClasses[bbox[5]]}</text>);
                texts.push(<text x={(bbox[0] * 100) + "%"}
                                y={(bbox[1] * 100) + "%"}
                                fontSize="8"
                                transform="translate(0 10)"
                                key={rand + "_textbg"}>{COCOClasses[bbox[5]]}</text>);
            }
        }
        else if (data["output_type"] == "classes") {
            let label = dm.data.class_map[data["prediction"][0]];
            const rand = Math.random().toString().substr(2, 8);
            texts.push(<text x="0" y="0" fontSize="10"
                            style={{stroke: 'white', strokeWidth: '0.6em'}}
                            transform="translate(0 10)"
                            key={rand + "_text"}>{label}</text>);
            texts.push(<text x="0" y="0" fontSize="10"
                            transform="translate(0 10)"
                            key={rand + "_textbg"}>{label}</text>);
        }

        return <>
          <List.Item style={{ marginTop: '15px', marginBottom: '0' }}>
            <div style={itemStyle}>
                <img style={style} src={new URL(`images/${data["id"]}`, dm.currentUrl).toString()} />
                <svg width="100%" height="100%" style={{position: "absolute", zIndex: "100", left: 0, top: 0}}>
                    {rects}
                    {texts}
                    Sorry, your browser does not support inline SVG.  
                </svg>
            </div>
          </List.Item>
        </>;
      }}
    />
  </>
});

export default RenderImages;