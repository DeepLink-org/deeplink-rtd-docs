// 检查当前页面的 URL
if (window.location.pathname.includes("op_download.html")) {
  // 获取表格元素
  var table = document.getElementsByTagName('table')[0];

  // 获取表格中的所有行
  var rows = table.getElementsByTagName('tr');

  // 计算表格总行数和总页数
  var rowCount = rows.length - 1; // 减去表头行
  var pageCount = Math.ceil(rowCount / 15); // 每页显示15行

  // 创建页码链接
  var pagination = document.createElement('div');
  pagination.className = 'pagination';

  for (var i = 1; i <= pageCount; i++) {
    var link = document.createElement('a');
    link.href = '#';
    link.innerHTML = i;
    link.onclick = function() {
      // 获取当前页码
      var currentPage = parseInt(this.innerHTML);

      // 计算当前页的起始行和结束行
      var startRow = (currentPage - 1) * 20 + 1; // 加上表头行
      var endRow = Math.min(currentPage * 20, rowCount) + 1; // 加上表头行

      // 隐藏所有行
      for (var j = 1; j < rows.length; j++) {
        rows[j].style.display = 'none';
      }

      // 显示当前页的行
      for (var j = startRow; j < endRow; j++) {
        rows[j].style.display = '';
      }

      // 更新页码链接的样式
      var links = pagination.getElementsByTagName('a');
      for (var j = 0; j < links.length; j++) {
        links[j].classList.remove('active');
      }
      this.classList.add('active');

      // 阻止链接的默认行为
      return false;
    };

    pagination.appendChild(link);
  }

  // 将页码链接添加到页面中
  table.parentNode.insertBefore(pagination, table.nextSibling);

  // 默认显示第一页
  pagination.getElementsByTagName('a')[0].click();

  // 下载时显示确认提示信息
  function showConfirmation(event) {
      event.preventDefault();
      if (confirm("确定要下载算子图谱文件吗？")) {
        var url = event.target.href;
        var xhr = new XMLHttpRequest();
        xhr.open('GET', url, true);
        xhr.responseType = 'blob';
        xhr.onload = function() {
          if (this.status === 200) {
            var blob = new Blob([this.response], {type: 'application/octet-stream'});
            var link = document.createElement('a');
            link.href = window.URL.createObjectURL(blob);
            link.download = url.substring(url.lastIndexOf('/')+1);
            link.click();
          }
        };
        xhr.send();
      }
    }

  $(document).ready(function() {
      // 获取下拉菜单和筛选按钮
      var field1Select = $("#field1-select");
      var field2Select = $("#field2-select");
      var filterButton = $("#filter-button");
    
      // 监听筛选按钮的点击事件
      filterButton.on("click", function() {
        // 获取选中的值
        var field1Value = field1Select.val();
        var field2Value = field2Select.val();
    
        // 获取包含表格的父元素，并找到其中的行
        var tableRows = $("table.docutils").find("tr");


    
        // 遍历表格行
        var filteredRows = [];
        tableRows.each(function(i, row) {
          var field1Cell = $(row).find("td:eq(0)");
          var field2Cell = $(row).find("td:eq(2)");
    
          // 获取行中的分类和分级值
          var field1 = field1Cell.text();
          var field2 = field2Cell.text();
    
          // 检查是否与筛选条件匹配
          var field1Match = field1Value === "" || field1 === field1Value;
          var field2Match = field2Value === "" || field2 === field2Value;
    
          // 根据匹配结果显示或隐藏行
          if (field1Match && field2Match) {
            filteredRows.push(row);
            $(row).show();
          } else {
            $(row).hide();
          }
          
          // 显示表头行
          var tableHeaderRow = $("table.docutils").find("tr:first");
          tableHeaderRow.show();
        });

        // 计算新的总行数和总页数
        var filteredRowCount = filteredRows.length;
        var filteredPageCount = Math.ceil(filteredRowCount / 15);

        // 更新页码链接的数量和显示内容
        pagination.innerHTML = "";
        for (var i = 1; i <= filteredPageCount; i++) {
          var link = document.createElement('a');
          link.href = '#';
          link.innerHTML = i;
          link.onclick = function() {
            // 获取当前页码
            var currentPage = parseInt(this.innerHTML);

            // 计算当前页的起始行和结束行
            var startRow = (currentPage - 1) * 20 + 1; // 加上表头行
            var endRow = Math.min(currentPage * 20, filteredRowCount) + 1; // 加上表头行

            // 隐藏所有行
            for (var j = 1; j < filteredRows.length; j++) {
              filteredRows[j].style.display = 'none';
            }

            // 显示当前页的行
            for (var j = startRow; j < endRow; j++) {
              filteredRows[j].style.display = '';
            }

            // 更新页码链接的样式
            var links = pagination.getElementsByTagName('a');
            for (var j = 0; j < links.length; j++) {
              links[j].classList.remove('active');
            }
            this.classList.add('active');

            // 阻止链接的默认行为
            return false;
          };

          pagination.appendChild(link);
        }

        // 默认显示第一页
        pagination.getElementsByTagName('a')[0].click();
      });
    });
}
