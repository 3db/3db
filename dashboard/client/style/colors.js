import  colormap from 'colormap';


export const CORRECT_COLOR = '#32a852';
export const INCORRECT_COLOR = '#a83232';


export const colors = colormap({
    colormap: [{index: 0, rgb: [168, 50, 50]}, {index: 1, rgb: [50, 168, 82]}],
    nshades: 101,
    format: 'hex',
    alpha: 1
});
