// plot.js - Iteration metrics plotting for OpenEvolve visualizer
import { getSelectedMetric } from './main.js';

let svg = null;
let g = null;
let zoomBehavior = null;
let lastTransform = null;
let iterationMetricsData = [];

// Initialize plot view
(function() {
    window.addEventListener('DOMContentLoaded', function() {
        const plotDiv = document.getElementById('view-plot');
        if (!plotDiv) return;

        // Listen for metric changes
        const metricSelect = document.getElementById('metric-select');
        if (metricSelect) {
            metricSelect.addEventListener('change', function() {
                if (plotDiv.style.display !== 'none') {
                    updatePlot(iterationMetricsData);
                }
            });
        }

        // Re-render when tab becomes active
        document.getElementById('tab-plot')?.addEventListener('click', function() {
            if (iterationMetricsData && iterationMetricsData.length > 0) {
                updatePlot(iterationMetricsData);
            }
        });

        // Responsive resize
        window.addEventListener('resize', function() {
            if (plotDiv.style.display !== 'none' && iterationMetricsData.length > 0) {
                updatePlot(iterationMetricsData);
            }
        });
    });
})();

export function renderPlot(data) {
    iterationMetricsData = data || [];

    if (!iterationMetricsData || iterationMetricsData.length === 0) {
        showNoDataMessage();
        return;
    }

    updatePlot(iterationMetricsData);
}

function showNoDataMessage() {
    const plotDiv = document.getElementById('view-plot');
    if (!plotDiv) return;

    plotDiv.innerHTML = `
        <div style="display:flex;align-items:center;justify-content:center;height:400px;color:#888;font-size:1.2em;">
            <div style="text-align:center;">
                <div style="font-size:3em;margin-bottom:0.5em;">📊</div>
                <div>No iteration metrics available</div>
                <div style="font-size:0.9em;margin-top:0.5em;">
                    Metrics are loaded from multiple checkpoint directories
                </div>
            </div>
        </div>
    `;
}

function updatePlot(data) {
    if (!data || data.length === 0) {
        showNoDataMessage();
        return;
    }

    const metric = getSelectedMetric();

    // Filter data points that have the selected metric
    const validData = data.filter(d =>
        d.metrics &&
        typeof d.metrics[metric] === 'number' &&
        isFinite(d.metrics[metric])
    );

    if (validData.length === 0) {
        const plotDiv = document.getElementById('view-plot');
        plotDiv.innerHTML = `
            <div style="display:flex;align-items:center;justify-content:center;height:400px;color:#888;font-size:1.2em;">
                <div style="text-align:center;">
                    <div style="font-size:3em;margin-bottom:0.5em;">⚠️</div>
                    <div>No data for metric: <strong>${metric}</strong></div>
                    <div style="font-size:0.9em;margin-top:0.5em;">
                        Select a different metric or check checkpoint data
                    </div>
                </div>
            </div>
        `;
        return;
    }

    // Get or create SVG
    if (!svg) {
        svg = d3.select('#plot-graph');
        if (svg.empty()) {
            svg = d3.select('#view-plot')
                .append('svg')
                .attr('id', 'plot-graph')
                .style('display', 'block');
        }
    }

    // Get or create group
    g = svg.select('g.zoom-group');
    if (g.empty()) {
        g = svg.append('g').attr('class', 'zoom-group');
    }

    // Setup zoom behavior
    if (!zoomBehavior) {
        zoomBehavior = d3.zoom()
            .scaleExtent([0.5, 10])
            .on('zoom', function(event) {
                g.attr('transform', event.transform);
                lastTransform = event.transform;
            });
        svg.call(zoomBehavior);
    }

    // Sizing
    const sidebarEl = document.getElementById('sidebar');
    const padding = 32;
    const windowWidth = window.innerWidth;
    const windowHeight = window.innerHeight;
    const toolbarHeight = document.getElementById('toolbar')?.offsetHeight || 60;
    const sidebarWidth = sidebarEl?.offsetWidth || 400;
    const width = Math.max(windowWidth - sidebarWidth - padding, 600);
    const height = Math.max(windowHeight - toolbarHeight - 80, 400);

    svg.attr('width', width).attr('height', height);

    // Clear previous content
    g.selectAll('*').remove();

    // Margins
    const margin = {top: 60, right: 60, bottom: 60, left: 80};
    const innerWidth = width - margin.left - margin.right;
    const innerHeight = height - margin.top - margin.bottom;

    // Scales
    const xExtent = d3.extent(validData, d => d.iteration);
    const yExtent = d3.extent(validData, d => d.metrics[metric]);

    const xScale = d3.scaleLinear()
        .domain(xExtent)
        .range([margin.left, margin.left + innerWidth])
        .nice();

    const yScale = d3.scaleLinear()
        .domain(yExtent)
        .range([margin.top + innerHeight, margin.top])
        .nice();

    // Add grid lines
    g.append('g')
        .attr('class', 'grid')
        .attr('transform', `translate(0,${margin.top + innerHeight})`)
        .style('stroke', '#e0e0e0')
        .style('stroke-opacity', 0.3)
        .call(d3.axisBottom(xScale)
            .tickSize(-innerHeight)
            .tickFormat('')
        );

    g.append('g')
        .attr('class', 'grid')
        .attr('transform', `translate(${margin.left},0)`)
        .style('stroke', '#e0e0e0')
        .style('stroke-opacity', 0.3)
        .call(d3.axisLeft(yScale)
            .tickSize(-innerWidth)
            .tickFormat('')
        );

    // Add axes
    const xAxis = g.append('g')
        .attr('class', 'axis')
        .attr('transform', `translate(0,${margin.top + innerHeight})`)
        .call(d3.axisBottom(xScale).ticks(10));

    const yAxis = g.append('g')
        .attr('class', 'axis')
        .attr('transform', `translate(${margin.left},0)`)
        .call(d3.axisLeft(yScale).ticks(10));

    // Add axis labels
    g.append('text')
        .attr('class', 'axis-label')
        .attr('x', margin.left + innerWidth / 2)
        .attr('y', height - 15)
        .attr('text-anchor', 'middle')
        .attr('font-size', '1.1em')
        .attr('fill', '#888')
        .text('Iteration');

    g.append('text')
        .attr('class', 'axis-label')
        .attr('transform', 'rotate(-90)')
        .attr('x', -(margin.top + innerHeight / 2))
        .attr('y', 20)
        .attr('text-anchor', 'middle')
        .attr('font-size', '1.1em')
        .attr('fill', '#888')
        .text(metric);

    // Add title
    g.append('text')
        .attr('class', 'plot-title')
        .attr('x', width / 2)
        .attr('y', 30)
        .attr('text-anchor', 'middle')
        .attr('font-size', '1.3em')
        .attr('font-weight', 'bold')
        .attr('fill', '#333')
        .text(`Evolution Progress: ${metric}`);

    // Line generator
    const line = d3.line()
        .x(d => xScale(d.iteration))
        .y(d => yScale(d.metrics[metric]))
        .curve(d3.curveMonotoneX);

    // Draw line
    g.append('path')
        .datum(validData)
        .attr('class', 'plot-line')
        .attr('fill', 'none')
        .attr('stroke', '#2196f3')
        .attr('stroke-width', 2.5)
        .attr('d', line);

    // Add data points
    const points = g.selectAll('.plot-point')
        .data(validData)
        .enter()
        .append('circle')
        .attr('class', 'plot-point')
        .attr('cx', d => xScale(d.iteration))
        .attr('cy', d => yScale(d.metrics[metric]))
        .attr('r', 5)
        .attr('fill', '#2196f3')
        .attr('stroke', '#fff')
        .attr('stroke-width', 2)
        .style('cursor', 'pointer');

    // Add tooltips
    const tooltip = d3.select('#view-plot')
        .append('div')
        .attr('class', 'plot-tooltip')
        .style('position', 'absolute')
        .style('visibility', 'hidden')
        .style('background', 'rgba(0,0,0,0.8)')
        .style('color', 'white')
        .style('padding', '8px 12px')
        .style('border-radius', '6px')
        .style('font-size', '0.9em')
        .style('pointer-events', 'none')
        .style('z-index', 1000);

    points
        .on('mouseover', function(event, d) {
            d3.select(this)
                .transition()
                .duration(100)
                .attr('r', 8)
                .attr('stroke-width', 3);

            tooltip
                .style('visibility', 'visible')
                .html(`
                    <strong>Iteration ${d.iteration}</strong><br/>
                    ${metric}: ${d.metrics[metric].toFixed(4)}
                `);
        })
        .on('mousemove', function(event) {
            tooltip
                .style('top', (event.pageY - 50) + 'px')
                .style('left', (event.pageX + 15) + 'px');
        })
        .on('mouseout', function() {
            d3.select(this)
                .transition()
                .duration(100)
                .attr('r', 5)
                .attr('stroke-width', 2);

            tooltip.style('visibility', 'hidden');
        });

    // Auto-zoom to fit if first render
    if (!lastTransform) {
        const bounds = g.node().getBBox();
        const fullWidth = width;
        const fullHeight = height;

        const scale = 0.9 * Math.min(
            fullWidth / bounds.width,
            fullHeight / bounds.height
        );

        const tx = (fullWidth - scale * (bounds.x + bounds.width / 2)) / 2;
        const ty = (fullHeight - scale * (bounds.y + bounds.height / 2)) / 2;

        const transform = d3.zoomIdentity
            .translate(tx, ty)
            .scale(Math.min(scale, 1));

        svg.call(zoomBehavior.transform, transform);
        lastTransform = transform;
    } else {
        // Reapply last transform
        svg.call(zoomBehavior.transform, lastTransform);
    }
}

// Export for use in main.js
window.renderPlot = renderPlot;
