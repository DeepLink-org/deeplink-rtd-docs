## 算子图谱下载

<script type="text/javascript" src="../../_static/custom.js"></script>


<style>
  .button-container {
    display: flex;
    justify-content: space-between;
    margin-top: 10px;
  }
</style>

<div>
  <label for="field1-select">标准分类：</label>
  <select id="field1-select">
    <option value="">全部</option>
    <option value="Convolution">Convolution</option>
    <option value="Linear">Linear</option>
    <option value="Pooling">Pooling</option>
    <option value="Pad">Pad</option>
    <option value="Loss">Loss</option>
    <option value="Norm">Norm</option>
    <option value="Activation">Activation</option>
    <option value="Dropout">Dropout</option>
    <option value="Interpolate">Interpolate</option>
    <option value="BLAS">BLAS</option>
    <option value="Linalg">Linalg</option>
    <option value="Permute">Permute</option>
    <option value="View">View</option>
    <option value="Advanced Indexing">Advanced Indexing</option>
    <option value="Distribution">Distribution</option>
    <option value="Sort">Sort</option>
    <option value="Element-wise">Element-wise</option>
    <option value="Broadcast">Broadcast</option>
    <option value="Reduce">Reduce</option>
    <option value="Composite">Composite</option>
    <option value="Misc">Misc</option>
  </select>
</div>

<div>
  <label for="field2-select">算子分级：</label>
  <select id="field2-select">
    <option value="">全部</option>
    <option value="P0">P0</option>
    <option value="P1">P1</option>
    <option value="P2">P2</option>
  </select>
</div>


<a>
<button id="filter-button">筛选</button>
</a>

<a href="../../../../doc/Operators/operators.xlsx" target="_blank" class="button" onclick="showConfirmation(event)">
  导出数据<span class="icon">&#x2B07;</span> 
</a>


```{csv-table}
:header-rows: 0
:file: "./processed_op.csv"
```
