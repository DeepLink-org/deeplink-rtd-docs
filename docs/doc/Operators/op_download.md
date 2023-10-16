# 算子图谱下载


<a>
  <label for="field1-select">标准分类</label>
  <select id="field1-select">
    <option value="">请选择</option>
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
</a>

<a>
  <label for="field2-select">算子分级</label>
  <select id="field2-select">
    <option value="">请选择</option>
    <option value="P0">P0</option>
    <option value="P1">P1</option>
    <option value="P2">P2</option>
  </select>
</a>


<a>
<button id="filter-button" class="button">筛选数据</button>
</a>

<a href="../../../../doc/Operators/operators.xlsx" target="_blank" class="button" onclick="showConfirmation(event)">
  导出数据
</a>

<style>
/* 设置下拉菜单和筛选按钮的外观 */
#field2-select {
  margin-right: 20px;
  margin-bottom: 20px;
}

#field1-select {
  margin-right: 20px;
}

#filter-button {
  margin-right: 250px;
}

.button {
    /* -webkit-transition-duration: 0.4s; /* Safari 
    transition-duration: 0.4s; */
  background-color: #2980b9;
  border: none;
  color: white;
  padding: 5px 20px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 14px;
  margin: 2px 0px;
  cursor: pointer;
}
 
.button:hover {
    background-color: #008CBA; /* Green */
    color: white;
}


</style>


```{csv-table}
:header-rows: 1
:file: "./processed_op.csv"
```

<style>
table {
  table-layout: auto;
  width: 100%;
}

th,
td {
  text-align: left;
}

/* 设置表头单元格的最小宽度 */
th {
  white-space: nowrap;
  min-width: 150px;
  font-weight: bold;
  font-family: Arial, Helvetica, sans-serif;
  vertical-align: middle; /* 文字垂直居中 */
}

/* 设置表格内容单元格的最小宽度 */
td {
  white-space: nowrap;
  min-width: 150px;
}
</style>
